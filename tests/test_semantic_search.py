# ABOUTME: Unit tests for semantic search functionality and paper section extraction
# ABOUTME: Tests embeddings, similarity search, chunking, and content analysis capabilities
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.zotero_mcp_server.semantic_search import (
    SemanticSearchEngine,
    PaperSection,
    PaperChunk,
    _CHUNK_TOKENIZER,
)
from src.zotero_mcp_server.zotero_client import ZoteroItem


def make_encode_mock(fixed_calls=None):
    """Build a MagicMock for model.encode that sizes its output to match the input.

    fixed_calls maps a 1-indexed call number to the exact array to return for that call
    (used when a test needs specific, deterministic embedding values, e.g. for a query).
    Any call not listed gets dummy vectors shaped (len(texts), 3), values irrelevant.
    This mirrors how index_papers() now makes two encode() calls per index (document-level,
    then chunk-level), so a single fixed-size mock would silently under/over-fill the FAISS
    index relative to the number of chunks actually produced.
    """
    fixed_calls = fixed_calls or {}
    state = {"n": 0}

    def _encode(texts, **kwargs):
        state["n"] += 1
        if state["n"] in fixed_calls:
            return fixed_calls[state["n"]]
        n = len(texts) if isinstance(texts, list) else 1
        return np.tile(np.array([0.1, 0.2, 0.3], dtype=np.float32), (n, 1))

    return MagicMock(side_effect=_encode)


@pytest.fixture
def search_engine(tmp_path):
    """Create a SemanticSearchEngine instance for testing."""
    with patch('src.zotero_mcp_server.semantic_search.SentenceTransformer') as mock_transformer_cls:
        mock_transformer_cls.return_value.get_sentence_embedding_dimension.return_value = 3
        engine = SemanticSearchEngine(data_dir=str(tmp_path))
        return engine


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        ZoteroItem(
            key="PAPER1",
            title="Machine Learning in Healthcare",
            creators=[{"firstName": "John", "lastName": "Doe"}],
            abstract="This paper explores the applications of machine learning in healthcare settings.",
            date="2023",
            item_type="journalArticle",
            tags=["machine learning", "healthcare"],
            full_text="Introduction\nMachine learning has revolutionized healthcare.\n\nMethodology\nWe used deep learning models.\n\nResults\nOur model achieved 95% accuracy."
        ),
        ZoteroItem(
            key="PAPER2",
            title="Neural Networks for Image Recognition",
            creators=[{"firstName": "Jane", "lastName": "Smith"}],
            abstract="A comprehensive study of neural networks for image recognition tasks.",
            date="2023",
            item_type="journalArticle",
            tags=["neural networks", "computer vision"],
            full_text="Abstract\nNeural networks are powerful tools.\n\nIntroduction\nImage recognition is a key task.\n\nConclusion\nOur approach shows promise."
        )
    ]


@pytest.mark.asyncio
async def test_index_papers(search_engine, sample_papers):
    """Test indexing papers for semantic search."""
    mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    search_engine.model.encode = make_encode_mock({1: mock_embeddings})

    await search_engine.index_papers(sample_papers)

    assert search_engine.faiss_index.ntotal == 2
    assert "PAPER1" in search_engine.faiss_paper_keys
    assert "PAPER2" in search_engine.faiss_paper_keys
    assert len(search_engine.paper_index) == 2

    # Chunk-level index is built alongside the document-level one
    assert search_engine.chunk_faiss_index.ntotal == len(search_engine.chunks)
    assert search_engine.chunk_faiss_index.ntotal > 0
    assert all(chunk.paper_key in {"PAPER1", "PAPER2"} for chunk in search_engine.chunks.values())


@pytest.mark.asyncio
async def test_search_by_topic(search_engine, sample_papers):
    """Test semantic search by topic."""
    # Index-time embeddings: PAPER1 points along the query direction, PAPER2 is orthogonal to it
    mock_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    query_embedding = np.array([[1.0, 0.0, 0.0]])
    # Call 1 = document-level embeddings, call 2 = chunk-level embeddings (dummy), call 3 = query
    search_engine.model.encode = make_encode_mock({1: mock_embeddings, 3: query_embedding})

    # Index papers
    await search_engine.index_papers(sample_papers)

    results = await search_engine.search_by_topic("machine learning", max_results=5)

    assert len(results) == 1  # Only PAPER1 meets the default min_similarity threshold
    assert results[0][0].key == "PAPER1"
    assert results[0][1] == pytest.approx(1.0)  # Cosine similarity score


@pytest.mark.asyncio
async def test_find_relevant_sections(search_engine, sample_papers):
    """Test finding relevant sections within a paper."""
    mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    search_engine.model.encode = make_encode_mock({1: mock_embeddings})

    # Index papers
    await search_engine.index_papers(sample_papers)

    paper = sample_papers[0]  # Paper with full text

    # Mock similarity calculation
    with patch('src.zotero_mcp_server.semantic_search.cosine_similarity') as mock_cosine:
        mock_cosine.return_value = np.array([[0.9, 0.7, 0.5]])  # Similarities for sections

        sections = await search_engine.find_relevant_sections(paper, "machine learning", max_sections=3)

        assert len(sections) > 0
        assert all(isinstance(section, PaperSection) for section in sections)
        assert sections[0].relevance_score >= sections[1].relevance_score if len(sections) > 1 else True


@pytest.mark.asyncio
async def test_save_and_load_index_round_trip(search_engine, sample_papers, tmp_path):
    """Test that saved FAISS indexes (document + chunk) and metadata can be reloaded without re-embedding."""
    mock_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    search_engine.model.encode = make_encode_mock({1: mock_embeddings})

    await search_engine.index_papers(sample_papers)
    expected_chunk_count = search_engine.chunk_faiss_index.ntotal
    search_engine.save_index()

    assert search_engine.index_path.exists()
    assert search_engine.meta_path.exists()
    assert search_engine.chunk_index_path.exists()
    assert search_engine.chunk_meta_path.exists()

    with patch('src.zotero_mcp_server.semantic_search.SentenceTransformer') as mock_transformer_cls:
        mock_transformer_cls.return_value.get_sentence_embedding_dimension.return_value = 3
        reloaded_engine = SemanticSearchEngine(data_dir=str(tmp_path))

    loaded = reloaded_engine.load_index()

    assert loaded is True
    assert reloaded_engine.faiss_index.ntotal == 2
    assert set(reloaded_engine.faiss_paper_keys) == {"PAPER1", "PAPER2"}
    assert reloaded_engine.paper_index["PAPER1"].title == "Machine Learning in Healthcare"
    assert reloaded_engine.tfidf_matrix is not None

    # Chunk-level index round-trips too
    assert reloaded_engine.chunk_faiss_index.ntotal == expected_chunk_count
    assert len(reloaded_engine.chunks) == expected_chunk_count
    assert all(chunk.paper_key in {"PAPER1", "PAPER2"} for chunk in reloaded_engine.chunks.values())


def test_load_index_returns_false_when_missing(search_engine):
    """Test that load_index reports no saved index instead of erroring."""
    assert search_engine.load_index() is False


def test_extract_sections(search_engine):
    """Test extracting sections from full text."""
    full_text = """Introduction
This is the introduction section.

Methodology
This describes our methods.

Results
Here are the results.

Conclusion
This is the conclusion."""

    sections = search_engine._extract_sections(full_text)

    assert len(sections) >= 1
    assert any("Introduction" in section["title"] for section in sections)
    assert any("Methodology" in section["title"] for section in sections)


def test_extract_sections_no_headers(search_engine):
    """Test extracting sections when no clear headers are found."""
    full_text = "This is just plain text without any section headers."

    sections = search_engine._extract_sections(full_text)

    assert len(sections) == 1
    assert sections[0]["title"] == "Full Text"
    assert sections[0]["content"] == full_text


@pytest.mark.asyncio
async def test_search_by_keywords(search_engine, sample_papers):
    """Test TF-IDF keyword search."""
    # Index papers (this will create the TF-IDF matrix)
    search_engine.model.encode = make_encode_mock({1: np.array([[0.1, 0.2, 0.9], [0.3, 0.4, 0.1]])})
    await search_engine.index_papers(sample_papers)

    # Mock TF-IDF operations
    with patch.object(search_engine.tfidf_vectorizer, 'transform') as mock_transform, \
         patch('src.zotero_mcp_server.semantic_search.cosine_similarity') as mock_cosine:

        mock_transform.return_value = np.array([[0.5, 0.3]])
        mock_cosine.return_value = np.array([[0.8, 0.4]])

        results = await search_engine.search_by_keywords("machine learning")

        assert len(results) == 2
        assert results[0][0].key == "PAPER1"  # Higher score should be first
        assert results[0][1] >= results[1][1]  # Scores should be descending


def test_chunk_paper_splits_long_section_with_overlap(search_engine):
    """Test that a long section is split into multiple overlapping token chunks."""
    search_engine.chunk_size_tokens = 10
    search_engine.chunk_overlap_tokens = 3

    long_text = "word " * 100  # far more than 10 tokens, no section headers
    paper = ZoteroItem(
        key="LONGPAPER",
        title="Long Paper",
        creators=[],
        item_type="journalArticle",
        full_text=long_text,
    )

    chunks = search_engine._chunk_paper(paper)

    assert len(chunks) > 1
    assert all(isinstance(chunk, PaperChunk) for chunk in chunks)
    assert all(chunk.paper_key == "LONGPAPER" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    # Consecutive chunks should overlap by ~chunk_overlap_tokens tokens
    first_tokens = _CHUNK_TOKENIZER.encode(chunks[0].text)
    second_tokens = _CHUNK_TOKENIZER.encode(chunks[1].text)
    assert first_tokens[-3:] == second_tokens[:3]


def test_chunk_paper_short_sections_yield_one_chunk_each(search_engine):
    """Test that sections shorter than chunk_size_tokens each produce exactly one chunk."""
    paper = ZoteroItem(
        key="PAPER1",
        title="Machine Learning in Healthcare",
        creators=[],
        item_type="journalArticle",
        full_text="Introduction\nMachine learning has revolutionized healthcare.\n\nMethodology\nWe used deep learning models.\n\nResults\nOur model achieved 95% accuracy."
    )

    chunks = search_engine._chunk_paper(paper)
    sections = search_engine._extract_sections(paper.full_text)

    assert len(chunks) == len(sections)
    assert [chunk.section_title for chunk in chunks] == [s["title"] for s in sections]


def test_chunk_paper_falls_back_to_abstract_when_no_full_text(search_engine):
    """Test that papers without full text get a single Abstract chunk instead of being skipped."""
    paper = ZoteroItem(
        key="NOFULLTEXT",
        title="No Full Text Paper",
        creators=[{"lastName": "Doe"}],
        abstract="A short abstract with no full-text attachment available.",
        item_type="journalArticle",
    )

    chunks = search_engine._chunk_paper(paper)

    assert len(chunks) == 1
    assert chunks[0].section_title == "Abstract"
    assert chunks[0].paper_key == "NOFULLTEXT"
    assert "short abstract" in chunks[0].text


@pytest.mark.asyncio
async def test_search_chunks(search_engine):
    """Test that search_chunks retrieves the right chunk and reports its paper/section."""
    papers = [
        ZoteroItem(
            key="CHUNKPAPER1",
            title="Chunk Paper One",
            creators=[],
            item_type="journalArticle",
            full_text="this is a short plain paragraph with no section headers describing paper one.",
        ),
        ZoteroItem(
            key="CHUNKPAPER2",
            title="Chunk Paper Two",
            creators=[],
            item_type="journalArticle",
            full_text="this is a short plain paragraph with no section headers describing paper two.",
        ),
    ]

    doc_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    chunk_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # one chunk per paper
    query_embedding = np.array([[1.0, 0.0, 0.0]])
    search_engine.model.encode = make_encode_mock({1: doc_embeddings, 2: chunk_embeddings, 3: query_embedding})

    await search_engine.index_papers(papers)
    assert search_engine.chunk_faiss_index.ntotal == 2

    results = await search_engine.search_chunks("paper one topic", max_results=5, min_similarity=0.0)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.paper_key == "CHUNKPAPER1"
    assert top_score == pytest.approx(1.0)
    assert top_chunk.section_title == "Full Text"
