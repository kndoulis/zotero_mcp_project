# ABOUTME: Unit tests for the Zotero API client functionality
# ABOUTME: Tests authentication, item fetching, search, and full-text retrieval
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.zotero_mcp_server.zotero_client import ZoteroClient, ZoteroItem


@pytest.fixture
def zotero_client():
    """Create a ZoteroClient instance for testing."""
    return ZoteroClient(api_key="test_key", user_id="12345")


@pytest.fixture
def mock_response_data():
    """Mock response data from Zotero API."""
    return [
        {
            "key": "ABCD1234",
            "data": {
                "key": "ABCD1234",
                "title": "Test Paper Title",
                "creators": [
                    {"firstName": "John", "lastName": "Doe", "creatorType": "author"}
                ],
                "abstractNote": "This is a test abstract",
                "date": "2023",
                "itemType": "journalArticle",
                "collections": ["COLL123"],
                "tags": [{"tag": "machine learning"}, {"tag": "AI"}],
                "DOI": "10.1234/test.doi"
            }
        }
    ]


@pytest.mark.asyncio
async def test_get_library_items(zotero_client, mock_response_data):
    """Test fetching library items."""
    with patch.object(zotero_client.session, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        items = await zotero_client.get_library_items()
        
        assert len(items) == 1
        assert items[0].title == "Test Paper Title"
        assert items[0].key == "ABCD1234"
        assert len(items[0].creators) == 1
        assert items[0].creators[0]["firstName"] == "John"


@pytest.mark.asyncio
async def test_search_items(zotero_client, mock_response_data):
    """Test searching items by query."""
    with patch.object(zotero_client.session, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        items = await zotero_client.search_items("machine learning")
        
        assert len(items) == 1
        assert items[0].title == "Test Paper Title"
        
        # Verify the correct API call was made
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert "q=machine+learning" in call_args


@pytest.mark.asyncio
async def test_get_item_fulltext(zotero_client):
    """Test fetching full-text content."""
    with patch.object(zotero_client.session, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": "This is the full text content"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        fulltext = await zotero_client.get_item_fulltext("ABCD1234")
        
        assert fulltext == "This is the full text content"


@pytest.mark.asyncio
async def test_get_item_fulltext_not_available(zotero_client):
    """Test handling when full-text is not available."""
    with patch.object(zotero_client.session, 'get') as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError("404 Not Found", request=None, response=None)
        
        fulltext = await zotero_client.get_item_fulltext("ABCD1234")
        
        assert fulltext is None


@pytest.mark.asyncio
async def test_get_all_items_with_fulltext_uses_pdf_attachment_key(zotero_client, mock_response_data):
    """Full-text must be fetched using the PDF attachment's item key, not the parent's."""
    children_data = [
        {"key": "PDFKEY01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
        {"key": "HTMLKEY1", "data": {"itemType": "attachment", "contentType": "text/html"}},
    ]

    with patch.object(zotero_client, "get_library_items", new=AsyncMock(side_effect=[
        [ZoteroItem(key="ABCD1234", title="Test Paper Title", creators=[], item_type="journalArticle")],
        [],
    ])), \
         patch.object(zotero_client, "get_item_children", new=AsyncMock(return_value=children_data)) as mock_children, \
         patch.object(zotero_client, "get_item_fulltext", new=AsyncMock(return_value="the real paper text")) as mock_fulltext:

        items = await zotero_client.get_all_items_with_fulltext()

    assert len(items) == 1
    assert items[0].full_text == "the real paper text"
    mock_children.assert_awaited_once_with("ABCD1234")
    mock_fulltext.assert_awaited_once_with("PDFKEY01")  # attachment key, not "ABCD1234"


@pytest.mark.asyncio
async def test_get_all_items_with_fulltext_handles_no_pdf_attachment(zotero_client):
    """Items with no PDF attachment (e.g. only an HTML snapshot) keep full_text as None."""
    children_data = [{"key": "HTMLKEY1", "data": {"itemType": "attachment", "contentType": "text/html"}}]

    with patch.object(zotero_client, "get_library_items", new=AsyncMock(side_effect=[
        [ZoteroItem(key="NOATTACH1", title="No PDF Paper", creators=[], item_type="journalArticle")],
        [],
    ])), \
         patch.object(zotero_client, "get_item_children", new=AsyncMock(return_value=children_data)), \
         patch.object(zotero_client, "get_item_fulltext", new=AsyncMock(return_value="should not be called")) as mock_fulltext:

        items = await zotero_client.get_all_items_with_fulltext()

    assert items[0].full_text is None
    mock_fulltext.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_all_items_with_fulltext_continues_when_children_fetch_fails(zotero_client):
    """A children-fetch failure for one item shouldn't drop that item or crash the whole fetch."""
    with patch.object(zotero_client, "get_library_items", new=AsyncMock(side_effect=[
        [ZoteroItem(key="BROKEN01", title="Broken Item", creators=[], item_type="journalArticle")],
        [],
    ])), \
         patch.object(zotero_client, "get_item_children", new=AsyncMock(side_effect=httpx.HTTPStatusError(
             "500 Server Error", request=None, response=None
         ))):

        items = await zotero_client.get_all_items_with_fulltext()

    assert len(items) == 1
    assert items[0].full_text is None


def test_parse_item(zotero_client):
    """Test parsing Zotero API item data."""
    data = {
        "title": "Test Title",
        "creators": [{"firstName": "Jane", "lastName": "Smith"}],
        "abstractNote": "Test abstract",
        "date": "2023",
        "itemType": "journalArticle",
        "collections": ["COLL1"],
        "tags": [{"tag": "test"}],
        "DOI": "10.1234/test"
    }
    
    item = zotero_client._parse_item(data, "TEST123")
    
    assert item.key == "TEST123"
    assert item.title == "Test Title"
    assert item.abstract == "Test abstract"
    assert item.date == "2023"
    assert item.item_type == "journalArticle"
    assert item.doi == "10.1234/test"
    assert len(item.creators) == 1
    assert len(item.collections) == 1
    assert len(item.tags) == 1