# ABOUTME: Main MCP server for Zotero integration with tools for search, discovery, and research assistance
# ABOUTME: Provides semantic search, paper discovery, section identification, and summary generation capabilities
import asyncio
import os
from typing import Any, Dict, List, Optional, Union
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp.server.lowlevel.server import NotificationOptions
from dotenv import load_dotenv

from .zotero_client import ZoteroClient, ZoteroItem
from .semantic_search import SemanticSearchEngine
from .rag_generate import ask_library
from .llm_client import DEFAULT_MODEL


# Load environment variables
load_dotenv()

# Global variables for the server
server = Server("zotero-mcp-server")
zotero_client: Optional[ZoteroClient] = None
search_engine: Optional[SemanticSearchEngine] = None
indexed_papers: List[ZoteroItem] = []
initialization_lock = asyncio.Lock()
is_initialized = False


def safe_format_creators(creators: List[Dict[str, Any]], max_creators: int = 3) -> str:
    """Safely format creator names handling null/undefined values."""
    if not creators or not isinstance(creators, list):
        return "Unknown Author"
    
    formatted_creators = []
    for creator in creators[:max_creators]:
        if not creator or not isinstance(creator, dict):
            continue
        
        try:
            first_name = creator.get("firstName", "").strip() if creator.get("firstName") else ""
            last_name = creator.get("lastName", "").strip() if creator.get("lastName") else ""
            
            if first_name and last_name:
                formatted_creators.append(f"{first_name} {last_name}")
            elif last_name:
                formatted_creators.append(last_name)
            elif first_name:
                formatted_creators.append(first_name)
            elif creator.get("name"):
                # Handle institutional or single-name creators
                formatted_creators.append(creator.get("name", "").strip())
        except (AttributeError, TypeError):
            continue
    
    return ", ".join(formatted_creators) if formatted_creators else "Unknown Author"


def validate_paper_object(paper: Any, paper_key: str = "unknown") -> bool:
    """Validate that a paper object has the expected structure."""
    try:
        if not paper:
            print(f"[VALIDATION] Paper {paper_key} is None/empty")
            return False
        
        # Test basic attribute access
        if not hasattr(paper, 'key') or not hasattr(paper, 'title'):
            print(f"[VALIDATION] Paper {paper_key} missing key/title attributes")
            return False
        
        # Test key access
        key_val = paper.key
        title_val = paper.title
        
        # Check for corrupted creators
        if hasattr(paper, 'creators') and paper.creators is not None:
            if not isinstance(paper.creators, list):
                print(f"[VALIDATION] Paper {paper_key} has non-list creators: {type(paper.creators)}")
                return False
            # Test creator access
            for i, creator in enumerate(paper.creators[:2]):
                if creator is None:
                    continue
                if not isinstance(creator, dict):
                    print(f"[VALIDATION] Paper {paper_key} creator[{i}] is not dict: {type(creator)}")
                    return False
                # Test safe access to creator properties
                _ = creator.get("firstName", "")
                _ = creator.get("lastName", "")
        
        # Check for corrupted tags
        if hasattr(paper, 'tags') and paper.tags is not None:
            if not isinstance(paper.tags, list):
                print(f"[VALIDATION] Paper {paper_key} has non-list tags: {type(paper.tags)}")
                return False
            # Test tag access
            for i, tag in enumerate(paper.tags[:2]):
                if tag is None:
                    continue
                if not isinstance(tag, str):
                    print(f"[VALIDATION] Paper {paper_key} tag[{i}] is not string: {type(tag)}")
                    # Don't fail for this, just warn
        
        # Test serialization
        try:
            import json
            test_data = {
                "key": paper.key,
                "title": paper.title,
                "creators": paper.creators[:1] if paper.creators else [],
                "tags": paper.tags[:1] if paper.tags else []
            }
            json.dumps(test_data)
        except Exception as serial_error:
            print(f"[VALIDATION] Paper {paper_key} serialization failed: {serial_error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[VALIDATION] Paper {paper_key} validation failed: {e}")
        return False


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools for the Zotero MCP server."""
    return [
        Tool(
            name="search_papers_by_topic",
            description="Find papers in your Zotero library relevant to a specific topic using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic or question to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity threshold (0.0-1.0, default: 0.3)",
                        "default": 0.3
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="search_papers_by_keywords",
            description="Find papers using traditional keyword/TF-IDF search",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Keywords to search for in paper titles, abstracts, and content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["keywords"]
            }
        ),
        Tool(
            name="find_relevant_sections",
            description="Find the most relevant sections within a specific paper for a given topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_key": {
                        "type": "string",
                        "description": "The Zotero item key of the paper to analyze"
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic or question to find relevant sections for"
                    },
                    "max_sections": {
                        "type": "integer",
                        "description": "Maximum number of sections to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["paper_key", "topic"]
            }
        ),
        Tool(
            name="get_library_overview",
            description="Get an overview of your entire Zotero library with item counts and collections",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_collections": {
                        "type": "boolean",
                        "description": "Whether to include collection information (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="get_paper_details",
            description="Get detailed information about a specific paper including full metadata and abstract",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_key": {
                        "type": "string",
                        "description": "The Zotero item key of the paper"
                    },
                    "include_fulltext": {
                        "type": "boolean",
                        "description": "Whether to include full-text content if available (default: false)",
                        "default": False
                    }
                },
                "required": ["paper_key"]
            }
        ),
        Tool(
            name="suggest_papers_for_chapter",
            description="Find all papers relevant to a specific chapter or section you want to write, with section recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "chapter_topic": {
                        "type": "string",
                        "description": "Description of the chapter or section you want to write about"
                    },
                    "chapter_outline": {
                        "type": "string",
                        "description": "Optional: Brief outline or key points you want to cover",
                        "default": ""
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "Maximum number of papers to suggest (default: 15)",
                        "default": 15
                    }
                },
                "required": ["chapter_topic"]
            }
        ),
        Tool(
            name="refresh_library_index",
            description="Refresh the search index by re-fetching and re-indexing all papers from Zotero",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="ask_library",
            description="Ask a question and get a synthesized, cited answer generated from your Zotero library's content (retrieval-augmented generation)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask your library"
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of source chunks to retrieve and ground the answer in (default: 8)",
                        "default": 8
                    },
                    "model": {
                        "type": "string",
                        "description": f"Claude model to use for generation (default: {DEFAULT_MODEL})",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": ["query"]
            }
        )
    ]


def debug_global_state(stage: str, tool_name: str):
    """Debug global state at different stages."""
    print(f"[STATE-DEBUG] {stage} - Tool: {tool_name}")
    print(f"  is_initialized: {is_initialized}")
    print(f"  zotero_client exists: {zotero_client is not None}")
    if zotero_client:
        print(f"  session closed: {getattr(zotero_client, '_session_closed', 'unknown')}")
        print(f"  session object: {type(zotero_client.session)}")
    print(f"  search_engine exists: {search_engine is not None}")
    print(f"  indexed_papers count: {len(indexed_papers) if indexed_papers else 0}")
    if indexed_papers:
        # Check first few papers for corruption
        for i, paper in enumerate(indexed_papers[:3]):
            try:
                # Try to access basic properties
                key = paper.key
                title = paper.title[:50] if paper.title else "No title"
                creators_type = type(paper.creators)
                tags_type = type(paper.tags)
                print(f"  paper[{i}]: key={key}, title='{title}', creators={creators_type}, tags={tags_type}")
            except Exception as e:
                print(f"  paper[{i}]: CORRUPTED - {e}")
    print(f"[STATE-DEBUG] {stage} complete\n")


def debug_return(tool_name: str, result: List[TextContent]) -> List[TextContent]:
    """Debug state before returning from tool."""
    debug_global_state("PRE-RETURN", tool_name)
    try:
        # Test result serialization
        result_text = result[0].text if result else ""
        print(f"[RESULT-DEBUG] Tool {tool_name} result length: {len(result_text)}")
        print(f"[RESULT-DEBUG] Result type: {type(result[0]) if result else 'None'}")
    except Exception as e:
        print(f"[RESULT-DEBUG] Error accessing result: {e}")
    return result


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for the Zotero MCP server."""
    global zotero_client, search_engine, indexed_papers, is_initialized
    
    debug_global_state("ENTRY", name)
    
    # Initialize clients if needed with proper synchronization
    async with initialization_lock:
        if not is_initialized or zotero_client is None or search_engine is None:
            print(f"[INIT-DEBUG] Initializing for {name}")
            await initialize_clients()
            debug_global_state("POST-INIT", name)
    
    try:
        if name == "search_papers_by_topic":
            topic = arguments["topic"]
            max_results = arguments.get("max_results", 10)
            min_similarity = arguments.get("min_similarity", 0.3)
            
            results = await search_engine.search_by_topic(topic, max_results, min_similarity)
            
            if not results:
                return debug_return(name, [TextContent(
                    type="text",
                    text=f"No papers found matching the topic '{topic}' with similarity >= {min_similarity}"
                )])
            
            response_text = f"Found {len(results)} papers relevant to '{topic}':\n\n"
            for i, (paper, similarity) in enumerate(results, 1):
                creators_str = safe_format_creators(paper.creators, 3)
                response_text += f"{i}. **{paper.title}** (Similarity: {similarity:.3f})\n"
                response_text += f"   Authors: {creators_str}\n"
                response_text += f"   Year: {paper.date}\n"
                response_text += f"   Type: {paper.item_type}\n"
                response_text += f"   Key: {paper.key}\n"
                if paper.abstract:
                    abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                    response_text += f"   Abstract: {abstract_preview}\n"
                response_text += "\n"
            
            return debug_return(name, [TextContent(type="text", text=response_text)])
        
        elif name == "search_papers_by_keywords":
            keywords = arguments["keywords"]
            max_results = arguments.get("max_results", 10)
            
            results = await search_engine.search_by_keywords(keywords, max_results)
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No papers found matching keywords '{keywords}'"
                )]
            
            response_text = f"Found {len(results)} papers matching keywords '{keywords}':\n\n"
            for i, (paper, score) in enumerate(results, 1):
                creators_str = safe_format_creators(paper.creators, 3)
                response_text += f"{i}. **{paper.title}** (Score: {score:.3f})\n"
                response_text += f"   Authors: {creators_str}\n"
                response_text += f"   Year: {paper.date}\n"
                response_text += f"   Key: {paper.key}\n\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "find_relevant_sections":
            paper_key = arguments["paper_key"]
            topic = arguments["topic"]
            max_sections = arguments.get("max_sections", 5)
            
            # Find the paper
            paper = next((p for p in indexed_papers if p.key == paper_key), None)
            if not paper:
                return [TextContent(
                    type="text",
                    text=f"Paper with key '{paper_key}' not found in the indexed library"
                )]
            
            # Validate paper object integrity
            if not validate_paper_object(paper, paper_key):
                return [TextContent(
                    type="text",
                    text=f"Paper with key '{paper_key}' has corrupted data. Please refresh the library index."
                )]
            
            sections = await search_engine.find_relevant_sections(paper, topic, max_sections)
            
            if not sections:
                return [TextContent(
                    type="text",
                    text=f"No relevant sections found in '{paper.title}' for topic '{topic}'"
                )]
            
            response_text = f"Found {len(sections)} relevant sections in '{paper.title}' for topic '{topic}':\n\n"
            for i, section in enumerate(sections, 1):
                content_preview = section.content[:300] + "..." if len(section.content) > 300 else section.content
                response_text += f"{i}. **{section.section_title}** (Relevance: {section.relevance_score:.3f})\n"
                response_text += f"   Content preview: {content_preview.strip()}\n\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "get_library_overview":
            include_collections = arguments.get("include_collections", True)
            
            # Get basic stats
            total_papers = len(indexed_papers)
            papers_with_fulltext = sum(1 for p in indexed_papers if p.full_text)
            
            # Get item type distribution
            item_types = {}
            for paper in indexed_papers:
                item_type = paper.item_type
                item_types[item_type] = item_types.get(item_type, 0) + 1
            
            response_text = f"**Zotero Library Overview**\n\n"
            response_text += f"Total items: {total_papers}\n"
            response_text += f"Items with full-text: {papers_with_fulltext}\n\n"
            response_text += f"**Item Types:**\n"
            for item_type, count in sorted(item_types.items()):
                response_text += f"- {item_type}: {count}\n"
            
            if include_collections:
                collections = await zotero_client.get_collections()
                response_text += f"\n**Collections:** {len(collections)}\n"
                for collection in collections[:10]:  # Show first 10
                    response_text += f"- {collection.name}\n"
                if len(collections) > 10:
                    response_text += f"... and {len(collections) - 10} more\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "get_paper_details":
            paper_key = arguments["paper_key"]
            include_fulltext = arguments.get("include_fulltext", False)
            
            paper = next((p for p in indexed_papers if p.key == paper_key), None)
            if not paper:
                return [TextContent(
                    type="text",
                    text=f"Paper with key '{paper_key}' not found"
                )]
            
            # Validate paper object integrity
            if not validate_paper_object(paper, paper_key):
                return [TextContent(
                    type="text",
                    text=f"Paper with key '{paper_key}' has corrupted data. Please refresh the library index."
                )]
            
            creators_str = safe_format_creators(paper.creators)
            
            response_text = f"**{paper.title}**\n\n"
            response_text += f"**Authors:** {creators_str}\n"
            response_text += f"**Year:** {paper.date}\n"
            response_text += f"**Type:** {paper.item_type}\n"
            response_text += f"**Key:** {paper.key}\n"
            
            if paper.doi:
                response_text += f"**DOI:** {paper.doi}\n"
            if paper.url:
                response_text += f"**URL:** {paper.url}\n"
            
            if paper.tags:
                try:
                    tag_str = ', '.join(paper.tags)
                    response_text += f"**Tags:** {tag_str}\n"
                except Exception:
                    response_text += f"**Tags:** [Error processing tags]\n"
            
            if paper.abstract:
                response_text += f"\n**Abstract:**\n{paper.abstract}\n"
            
            if include_fulltext and paper.full_text:
                fulltext_preview = paper.full_text[:1000] + "..." if len(paper.full_text) > 1000 else paper.full_text
                response_text += f"\n**Full-text (preview):**\n{fulltext_preview}\n"
            
            return debug_return(name, [TextContent(type="text", text=response_text)])
        
        elif name == "suggest_papers_for_chapter":
            chapter_topic = arguments["chapter_topic"]
            chapter_outline = arguments.get("chapter_outline", "")
            max_papers = arguments.get("max_papers", 15)
            
            # Combine topic and outline for search
            search_query = chapter_topic
            if chapter_outline:
                search_query += " " + chapter_outline
            
            # Search using semantic similarity
            results = await search_engine.search_by_topic(search_query, max_papers, min_similarity=0.2)
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No papers found relevant to the chapter topic '{chapter_topic}'"
                )]
            
            response_text = f"**Papers suggested for chapter: '{chapter_topic}'**\n\n"
            response_text += f"Found {len(results)} relevant papers:\n\n"
            
            for i, (paper, similarity) in enumerate(results, 1):
                creators_str = safe_format_creators(paper.creators, 2)
                
                response_text += f"**{i}. {paper.title}**\n"
                response_text += f"   Authors: {creators_str}\n"
                response_text += f"   Year: {paper.date} | Relevance: {similarity:.3f}\n"
                response_text += f"   Key: `{paper.key}`\n"
                
                if paper.abstract:
                    abstract_preview = paper.abstract[:150] + "..." if len(paper.abstract) > 150 else paper.abstract
                    response_text += f"   Abstract: {abstract_preview}\n"
                
                # Find most relevant sections if full-text is available
                if paper.full_text:
                    sections = await search_engine.find_relevant_sections(paper, chapter_topic, 2)
                    if sections:
                        response_text += f"   **Most relevant sections:**\n"
                        for section in sections:
                            response_text += f"   - {section.section_title} (relevance: {section.relevance_score:.2f})\n"
                
                response_text += "\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "refresh_library_index":
            await initialize_clients(force_refresh=True)
            return [TextContent(
                type="text",
                text=f"Library index refreshed! Indexed {len(indexed_papers)} papers."
            )]

        elif name == "ask_library":
            query = arguments["query"]
            max_chunks = arguments.get("max_chunks", 8)
            model = arguments.get("model", DEFAULT_MODEL)

            result = await ask_library(query, search_engine, max_chunks=max_chunks, model=model)

            response_text = f"**Answer:**\n{result['answer']}\n"
            if result["sources"]:
                response_text += "\n**Sources used:**\n"
                for source in result["sources"]:
                    response_text += (
                        f"- [{source['paper_key']}] {source['title']} "
                        f"(section: {source['section_title']}, relevance: {source['relevance']:.3f})\n"
                    )

            return debug_return(name, [TextContent(type="text", text=response_text)])

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR-DEBUG] Exception in tool {name}: {error_msg}")
        print(f"[ERROR-DEBUG] Exception type: {type(e).__name__}")
        debug_global_state("ERROR-STATE", name)
        
        # Check for session-specific errors that don't require full reinitialization
        session_errors = ["cannot convert undefined or null to object", "connection reset", "connection closed", "session closed", "httpx"]
        data_errors = ["json", "parse", "decode"]
        
        if any(phrase in error_msg.lower() for phrase in session_errors):
            try:
                print(f"Session error detected in {name}, attempting to refresh session...")
                # First try to just refresh the session without full reinitialization
                if zotero_client is not None:
                    await zotero_client.ensure_session_open()
                    # Retry the operation once with refreshed session
                    print(f"Retrying {name} with refreshed session...")
                    return await retry_tool_operation(name, arguments)
                else:
                    # If no client, do full reinitialization
                    async with initialization_lock:
                        is_initialized = False
                        await initialize_clients(force_refresh=True)
                    
                    print(f"Retrying {name} after full reinitialization...")
                    return await retry_tool_operation(name, arguments)
                
            except Exception as retry_error:
                retry_msg = str(retry_error)
                print(f"Retry failed for {name}: {retry_msg}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {error_msg}\nRetry attempt also failed: {retry_msg}\n\nThis may indicate an issue with your Zotero API credentials or library data. Please check your .env file and Zotero library."
                )]
        
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {error_msg}\n\nIf this error persists, please check your Zotero API credentials and library data."
        )]


async def retry_tool_operation(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Retry a tool operation after reinitializing clients."""
    global zotero_client, search_engine, indexed_papers
    
    if name == "search_papers_by_topic":
        topic = arguments["topic"]
        max_results = arguments.get("max_results", 10)
        min_similarity = arguments.get("min_similarity", 0.3)
        
        results = await search_engine.search_by_topic(topic, max_results, min_similarity)
        
        if not results:
            return [TextContent(
                type="text",
                text=f"No papers found matching the topic '{topic}' with similarity >= {min_similarity}"
            )]
        
        response_text = f"Found {len(results)} papers relevant to '{topic}':\n\n"
        for i, (paper, similarity) in enumerate(results, 1):
            creators_str = safe_format_creators(paper.creators, 3)
            response_text += f"{i}. **{paper.title}** (Similarity: {similarity:.3f})\n"
            response_text += f"   Authors: {creators_str}\n"
            response_text += f"   Year: {paper.date}\n"
            response_text += f"   Type: {paper.item_type}\n"
            response_text += f"   Key: {paper.key}\n"
            if paper.abstract:
                abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                response_text += f"   Abstract: {abstract_preview}\n"
            response_text += "\n"
        
        return [TextContent(type="text", text=response_text)]
    
    # Add other operation retries as needed
    return [TextContent(
        type="text",
        text=f"Retry not implemented for tool: {name}"
    )]


async def initialize_clients(force_refresh: bool = False) -> None:
    """Initialize Zotero client and search engine."""
    global zotero_client, search_engine, indexed_papers, is_initialized
    
    if not force_refresh and is_initialized and zotero_client is not None:
        return
    
    try:
        # Get API credentials
        api_key = os.getenv("ZOTERO_API_KEY")
        user_id = os.getenv("ZOTERO_USER_ID")
        
        if not api_key or not user_id:
            raise ValueError(
                "ZOTERO_API_KEY and ZOTERO_USER_ID must be set in environment variables. "
                "Copy .env.example to .env and fill in your credentials."
            )
        
        # Clean up existing client if any
        if zotero_client is not None:
            try:
                await zotero_client.close()
            except Exception:
                pass  # Ignore errors during cleanup
        
        # Initialize new clients
        zotero_client = ZoteroClient(api_key, user_id)
        search_engine = SemanticSearchEngine()

        # Reuse a persisted FAISS index when available, unless a refresh was requested
        if not force_refresh and search_engine.load_index():
            print(f"Loaded persisted index with {len(search_engine.paper_index)} papers.")
            indexed_papers = list(search_engine.paper_index.values())
        else:
            # Fetch and index papers without closing the session
            print("Fetching papers from Zotero library...")
            indexed_papers = await zotero_client.get_all_items_with_fulltext()

            print(f"Indexing {len(indexed_papers)} papers for semantic search...")
            await search_engine.index_papers(indexed_papers)
            search_engine.save_index()

        is_initialized = True
        print("Initialization completed successfully!")
        
    except Exception as e:
        is_initialized = False
        zotero_client = None
        search_engine = None
        indexed_papers = []
        raise e


async def cleanup():
    """Clean up resources when shutting down."""
    global zotero_client
    if zotero_client is not None:
        try:
            await zotero_client.close()
        except Exception:
            pass  # Ignore cleanup errors


async def main():
    """Main entry point for the MCP server."""
    try:
        # Run the server using stdio
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="zotero-mcp-server",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                ),
            )
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())