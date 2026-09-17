# ABOUTME: Semantic search engine for finding relevant papers using embeddings and similarity matching
# ABOUTME: Provides topic-based paper discovery and section identification within documents
import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel

from .zotero_client import ZoteroItem

_CHUNK_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class PaperSection(BaseModel):
    """Represents a section within a paper with its content and relevance score."""
    section_title: str
    content: str
    start_position: int
    end_position: int
    relevance_score: float


class PaperChunk(BaseModel):
    """A chunk of a paper's text, embedded separately for fine-grained retrieval."""
    chunk_id: str
    paper_key: str
    chunk_index: int
    section_title: str
    text: str


def _split_into_token_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks of ~chunk_size tokens (tiktoken cl100k_base)."""
    tokens = _CHUNK_TOKENIZER.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    stride = max(chunk_size - overlap, 1)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_CHUNK_TOKENIZER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride
    return chunks


class SemanticSearchEngine:
    """Semantic search engine for Zotero papers using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        data_dir: str = "data",
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
    ):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.paper_index: Dict[str, ZoteroItem] = {}
        # Inner-product index over L2-normalized vectors is equivalent to cosine similarity
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_paper_keys: List[str] = []  # row index -> paper key
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.tfidf_paper_keys: List[str] = []  # Keys corresponding to TF-IDF matrix rows

        # Chunk-level index: finer-grained than the document-level index above, used for RAG retrieval
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chunk_faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_faiss_keys: List[str] = []  # row index -> chunk_id
        self.chunks: Dict[str, PaperChunk] = {}

        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "paper_index.faiss"
        self.meta_path = self.data_dir / "paper_index_meta.json"
        self.chunk_index_path = self.data_dir / "chunk_index.faiss"
        self.chunk_meta_path = self.data_dir / "chunk_index_meta.json"

    def _build_paper_text(self, paper: ZoteroItem) -> Optional[str]:
        """Combine a paper's title, abstract, full-text preview, creators, and tags into one string."""
        text_parts = []
        if paper.title:
            text_parts.append(str(paper.title))

        if paper.abstract:
            text_parts.append(str(paper.abstract))

        if paper.full_text:
            # Limit full-text to first 2000 characters to avoid memory issues
            text_parts.append(str(paper.full_text[:2000]))

        # Add creator names and tags for better matching
        creator_names = []
        if isinstance(paper.creators, list):
            for creator in paper.creators:
                if not isinstance(creator, dict):
                    continue
                if creator.get('lastName'):
                    creator_names.append(str(creator['lastName']))
                if creator.get('firstName'):
                    creator_names.append(str(creator['firstName']))
                if creator.get('name'):  # Handle institutional creators
                    creator_names.append(str(creator['name']))

        if creator_names:
            text_parts.append(" ".join(creator_names))

        if paper.tags and isinstance(paper.tags, list):
            tag_strings = [str(tag) for tag in paper.tags if tag]
            if tag_strings:
                text_parts.append(" ".join(tag_strings))

        return " ".join(text_parts) if text_parts else None

    def _chunk_paper(self, paper: ZoteroItem) -> List[PaperChunk]:
        """Split a paper into overlapping, section-tagged chunks for fine-grained retrieval.

        Papers with full text are chunked per-section (reusing the same section extraction
        as find_relevant_sections), so each chunk keeps an accurate section label. Papers
        without full text fall back to a single chunk built from title/abstract/metadata.
        """
        if paper.full_text:
            sections = self._extract_sections(paper.full_text)
        else:
            fallback_text = self._build_paper_text(paper)
            sections = [{"title": "Abstract", "content": fallback_text}] if fallback_text else []

        chunks = []
        chunk_index = 0
        for section in sections:
            section_text = (section.get("content") or "").strip()
            if not section_text:
                continue
            for chunk_text in _split_into_token_chunks(
                section_text, self.chunk_size_tokens, self.chunk_overlap_tokens
            ):
                chunks.append(PaperChunk(
                    chunk_id=f"{paper.key}::{chunk_index}",
                    paper_key=paper.key,
                    chunk_index=chunk_index,
                    section_title=section["title"],
                    text=chunk_text,
                ))
                chunk_index += 1

        return chunks

    async def _index_chunks(self, papers: List[ZoteroItem]) -> None:
        """Build the chunk-level FAISS index used for fine-grained retrieval (e.g. RAG generation)."""
        all_chunks: List[PaperChunk] = []
        for paper in papers:
            if not paper or not hasattr(paper, 'key'):
                continue
            all_chunks.extend(self._chunk_paper(paper))

        self.chunk_faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_faiss_keys = []
        self.chunks = {}

        if not all_chunks:
            print("Chunk indexing complete! 0 chunks indexed.")
            return

        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        embeddings_array = np.asarray(chunk_embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        self.chunk_faiss_index.add(embeddings_array)

        for chunk in all_chunks:
            self.chunk_faiss_keys.append(chunk.chunk_id)
            self.chunks[chunk.chunk_id] = chunk

        print(f"Chunk indexing complete! {self.chunk_faiss_index.ntotal} chunks indexed across {len(papers)} papers.")

    async def index_papers(self, papers: List[ZoteroItem]) -> None:
        """Create embeddings for all papers in the library."""
        print(f"Indexing {len(papers)} papers for semantic search...")

        # Prepare text for embedding
        paper_texts = []
        paper_keys = []

        for paper in papers:
            if not paper or not hasattr(paper, 'key'):
                continue

            # Only add papers with meaningful content
            combined_text = self._build_paper_text(paper)
            if combined_text:
                paper_texts.append(combined_text)
                paper_keys.append(paper.key)

            # Store paper in index
            self.paper_index[paper.key] = paper

        # Create embeddings
        embeddings = self.model.encode(paper_texts, show_progress_bar=True)

        # Rebuild the FAISS index (supports re-indexing, e.g. on refresh_library_index)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_paper_keys = []

        if paper_texts:
            embeddings_array = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
            self.faiss_paper_keys = paper_keys.copy()

            # Create TF-IDF matrix for keyword-based search
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paper_texts)
            # Store the paper keys that correspond to TF-IDF matrix rows
            self.tfidf_paper_keys = paper_keys.copy()

        print(f"Indexing complete! {self.faiss_index.ntotal} papers indexed.")

        # Chunk-level index for fine-grained retrieval (Phase 2)
        await self._index_chunks(papers)

    def save_index(self) -> None:
        """Persist the FAISS indexes (document-level and chunk-level) and metadata to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(self.index_path))
        meta = {
            "faiss_paper_keys": self.faiss_paper_keys,
            "papers": {key: paper.model_dump() for key, paper in self.paper_index.items()},
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

        faiss.write_index(self.chunk_faiss_index, str(self.chunk_index_path))
        chunk_meta = {
            "chunk_faiss_keys": self.chunk_faiss_keys,
            "chunks": {chunk_id: chunk.model_dump() for chunk_id, chunk in self.chunks.items()},
        }
        with open(self.chunk_meta_path, "w") as f:
            json.dump(chunk_meta, f)

    def load_index(self) -> bool:
        """Load a previously persisted FAISS index and metadata, if present.

        Returns True if an index was loaded, False if none exists on disk yet.
        """
        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        self.faiss_index = faiss.read_index(str(self.index_path))
        with open(self.meta_path) as f:
            meta = json.load(f)

        self.faiss_paper_keys = meta["faiss_paper_keys"]
        self.paper_index = {key: ZoteroItem(**data) for key, data in meta["papers"].items()}

        # Rebuild TF-IDF from the same paper texts used to build the FAISS index.
        # This is cheap (no re-embedding) and keeps hybrid keyword search working after a load.
        paper_texts = [self._build_paper_text(self.paper_index[key]) for key in self.faiss_paper_keys]
        if paper_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paper_texts)
            self.tfidf_paper_keys = self.faiss_paper_keys.copy()

        if self.chunk_index_path.exists() and self.chunk_meta_path.exists():
            self.chunk_faiss_index = faiss.read_index(str(self.chunk_index_path))
            with open(self.chunk_meta_path) as f:
                chunk_meta = json.load(f)
            self.chunk_faiss_keys = chunk_meta["chunk_faiss_keys"]
            self.chunks = {chunk_id: PaperChunk(**data) for chunk_id, data in chunk_meta["chunks"].items()}
        else:
            # Index was persisted before chunk-level retrieval existed; chunks need a refresh to appear.
            self.chunk_faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_faiss_keys = []
            self.chunks = {}

        return True

    async def search_by_topic(self, topic: str, max_results: int = 10, min_similarity: float = 0.3) -> List[Tuple[ZoteroItem, float]]:
        """Find papers most relevant to a given topic."""
        if self.faiss_index.ntotal == 0:
            raise ValueError("No papers indexed. Call index_papers() first.")

        # Create embedding for the search topic
        topic_embedding = np.asarray(self.model.encode([topic]), dtype=np.float32)
        faiss.normalize_L2(topic_embedding)

        # Search the whole index so filtering by min_similarity doesn't truncate results early
        scores, indices = self.faiss_index.search(topic_embedding, self.faiss_index.ntotal)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score >= min_similarity:
                paper_key = self.faiss_paper_keys[idx]
                paper = self.paper_index[paper_key]
                results.append((paper, float(score)))

        # FAISS IndexFlatIP already returns matches sorted by score (descending)
        return results[:max_results]

    async def search_chunks(self, query: str, max_results: int = 10, min_similarity: float = 0.0) -> List[Tuple[PaperChunk, float]]:
        """Find the most relevant chunks across the library for a given query.

        Unlike search_by_topic (document-level), this returns individual chunks so callers
        can see exactly which passage/section a match came from - used by RAG generation.
        """
        if self.chunk_faiss_index.ntotal == 0:
            raise ValueError("No chunks indexed. Call index_papers() first.")

        query_embedding = np.asarray(self.model.encode([query]), dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.chunk_faiss_index.search(query_embedding, self.chunk_faiss_index.ntotal)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score >= min_similarity:
                chunk_id = self.chunk_faiss_keys[idx]
                results.append((self.chunks[chunk_id], float(score)))

        return results[:max_results]

    async def search_by_keywords(self, keywords: str, max_results: int = 10) -> List[Tuple[ZoteroItem, float]]:
        """Find papers using TF-IDF keyword matching."""
        if self.tfidf_matrix is None or not self.tfidf_paper_keys:
            raise ValueError("No papers indexed. Call index_papers() first.")
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([keywords])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[::-1][:max_results]
        
        results = []
        for idx in top_indices:
            if idx < len(self.tfidf_paper_keys) and similarities[idx] > 0:  # Bounds check + non-zero
                paper_key = self.tfidf_paper_keys[idx]  # Use the correct key mapping
                if paper_key in self.paper_index:  # Verify key exists
                    paper = self.paper_index[paper_key]
                    results.append((paper, float(similarities[idx])))
        
        return results
    
    async def find_relevant_sections(self, paper: ZoteroItem, topic: str, max_sections: int = 5) -> List[PaperSection]:
        """Find the most relevant sections within a paper for a given topic."""
        if not paper.full_text:
            return []
        
        # Split paper into sections
        sections = self._extract_sections(paper.full_text)
        if not sections:
            return []
        
        # Create embeddings for topic and sections
        topic_embedding = self.model.encode([topic])
        section_texts = [s["content"] for s in sections]
        section_embeddings = self.model.encode(section_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(topic_embedding, section_embeddings).flatten()
        
        # Create PaperSection objects with scores
        paper_sections = []
        for i, section in enumerate(sections):
            if similarities[i] > 0.2:  # Minimum relevance threshold
                paper_sections.append(PaperSection(
                    section_title=section["title"],
                    content=section["content"],
                    start_position=section["start"],
                    end_position=section["end"],
                    relevance_score=float(similarities[i])
                ))
        
        # Sort by relevance score
        paper_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return paper_sections[:max_sections]
    
    def _extract_sections(self, full_text: str) -> List[Dict[str, any]]:
        """Extract sections from full-text content."""
        # Common academic paper section patterns. Anchored with $ (not \n) since each pattern
        # is matched against a single stripped line, which never contains a newline itself.
        section_patterns = [
            r'^(\d+\.?\s*[A-Z][A-Za-z\s]{3,50})$',  # Numbered sections
            r'^([A-Z][A-Z\s]{3,50})$',  # All caps sections
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$',  # Title case sections
        ]

        sections = []
        current_section = {"title": "Introduction", "content": "", "start": 0}
        found_any_header = False

        lines = full_text.split('\n')
        position = 0

        for line in lines:
            line_with_newline = line + '\n'

            # Check if this line is a section header
            is_section_header = False
            section_title = None

            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    section_title = match.group(1).strip()
                    is_section_header = True
                    break

            if is_section_header and section_title:
                found_any_header = True
                # Save current section if it has content
                if current_section["content"].strip():
                    current_section["end"] = position
                    sections.append(current_section.copy())

                # Start new section
                current_section = {
                    "title": section_title,
                    "content": "",
                    "start": position
                }
            else:
                # Add line to current section
                current_section["content"] += line_with_newline

            position += len(line_with_newline)

        # Add the last section
        if current_section["content"].strip():
            current_section["end"] = position
            sections.append(current_section)

        # If no headers were ever detected, treat the entire text as one section rather than
        # keeping the "Introduction" placeholder title the loop started with.
        if not found_any_header:
            return [{
                "title": "Full Text",
                "content": full_text,
                "start": 0,
                "end": len(full_text)
            }]

        # Defensive fallback: headers were found but somehow no section survived (shouldn't
        # normally happen, since a detected header always starts a new section).
        if not sections:
            sections.append({
                "title": "Full Text",
                "content": full_text,
                "start": 0,
                "end": len(full_text)
            })

        return sections