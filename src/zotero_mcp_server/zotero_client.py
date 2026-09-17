# ABOUTME: Zotero Web API client for fetching library items, collections, and full-text content
# ABOUTME: Handles authentication, rate limiting, and data formatting for MCP server integration
import asyncio
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import httpx
from pydantic import BaseModel


class ZoteroItem(BaseModel):
    """Represents a Zotero library item with metadata and content."""
    key: str
    title: str
    creators: List[Dict[str, Any]]
    abstract: Optional[str] = None
    date: Optional[str] = None
    item_type: str
    collections: List[str] = []
    tags: List[str] = []
    url: Optional[str] = None
    doi: Optional[str] = None
    full_text: Optional[str] = None
    attachments: List[Dict[str, Any]] = []


class ZoteroCollection(BaseModel):
    """Represents a Zotero collection/folder."""
    key: str
    name: str
    parent_collection: Optional[str] = None


class ZoteroClient:
    """Async client for the Zotero Web API."""
    
    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key
        self.user_id = user_id
        self.base_url = "https://api.zotero.org"
        self.session = httpx.AsyncClient(
            headers={"Zotero-API-Key": api_key},
            timeout=30.0
        )
        self._session_closed = False
    
    async def ensure_session_open(self):
        """Ensure the session is open and ready for requests."""
        if self.session.is_closed or self._session_closed:
            print("Recreating closed httpx session...")
            self.session = httpx.AsyncClient(
                headers={"Zotero-API-Key": self.api_key},
                timeout=30.0
            )
            self._session_closed = False
    
    async def close(self):
        """Explicitly close the session."""
        if not self._session_closed and not self.session.is_closed:
            await self.session.aclose()
            self._session_closed = True
    
    async def get_collections(self) -> List[ZoteroCollection]:
        """Fetch all collections from the user's library."""
        await self.ensure_session_open()
        url = f"{self.base_url}/users/{self.user_id}/collections"
        response = await self.session.get(url)
        response.raise_for_status()
        
        try:
            response_data = response.json()
            if not isinstance(response_data, list):
                print(f"Warning: Expected list response, got {type(response_data)}")
                return []
        except (ValueError, TypeError) as e:
            print(f"Error parsing JSON response: {e}")
            return []
        
        collections = []
        for item in response_data:
            if not isinstance(item, dict):
                continue
            data = item.get("data", {})
            if not isinstance(data, dict):
                continue
            parent_collection = data.get("parentCollection")
            if parent_collection and isinstance(parent_collection, str):
                parent_collection = parent_collection
            else:
                parent_collection = None
                
            collections.append(ZoteroCollection(
                key=data.get("key", ""),
                name=data.get("name", ""),
                parent_collection=parent_collection
            ))
        
        return collections
    
    async def get_library_items(self, limit: int = 100, start: int = 0) -> List[ZoteroItem]:
        """Fetch items from the user's library."""
        await self.ensure_session_open()
        params = {
            "limit": limit,
            "start": start,
            "format": "json",
            "include": "data"
        }
        url = f"{self.base_url}/users/{self.user_id}/items?{urlencode(params)}"
        response = await self.session.get(url)
        response.raise_for_status()
        
        try:
            response_data = response.json()
            if not isinstance(response_data, list):
                print(f"Warning: Expected list response, got {type(response_data)}")
                return []
        except (ValueError, TypeError) as e:
            print(f"Error parsing JSON response: {e}")
            return []
        
        items = []
        for item_data in response_data:
            if not isinstance(item_data, dict):
                continue
            data = item_data.get("data", {})
            if not isinstance(data, dict):
                continue
            
            # Skip attachment items for now
            if data.get("itemType") == "attachment":
                continue
                
            items.append(self._parse_item(data, item_data.get("key", "")))
        
        return items
    
    async def search_items(self, query: str, limit: int = 50) -> List[ZoteroItem]:
        """Search items by title, creator, or other metadata."""
        await self.ensure_session_open()
        params = {
            "q": query,
            "limit": limit,
            "format": "json",
            "include": "data"
        }
        url = f"{self.base_url}/users/{self.user_id}/items?{urlencode(params)}"
        response = await self.session.get(url)
        response.raise_for_status()
        
        items = []
        for item_data in response.json():
            data = item_data.get("data", {})
            
            if data.get("itemType") == "attachment":
                continue
                
            items.append(self._parse_item(data, item_data.get("key", "")))
        
        return items
    
    async def get_item_children(self, item_key: str) -> List[Dict[str, Any]]:
        """Get child items (attachments) for a given item."""
        await self.ensure_session_open()
        url = f"{self.base_url}/users/{self.user_id}/items/{item_key}/children"
        response = await self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    async def get_item_fulltext(self, item_key: str) -> Optional[str]:
        """Get the full-text content of an item if available."""
        try:
            await self.ensure_session_open()
            url = f"{self.base_url}/users/{self.user_id}/items/{item_key}/fulltext"
            response = await self.session.get(url)
            response.raise_for_status()
            
            try:
                data = response.json()
                if not isinstance(data, dict):
                    print(f"Warning: Expected dict response for fulltext, got {type(data)}")
                    return None
                return data.get("content", "")
            except (ValueError, TypeError) as e:
                print(f"Error parsing fulltext JSON response: {e}")
                return None
        except httpx.HTTPStatusError:
            # Full-text not available for this item
            return None
    
    async def get_all_items_with_fulltext(self) -> List[ZoteroItem]:
        """Get all library items and attempt to fetch full-text from their PDF attachments.

        The Zotero fulltext endpoint only returns content for an attachment's own item key,
        not its parent bibliography item, so this fetches each item's children first to find
        a PDF attachment before requesting full-text.
        """
        all_items = []
        start = 0
        limit = 100

        while True:
            items = await self.get_library_items(limit=limit, start=start)
            if not items:
                break

            # Fetch full-text for each item via its PDF attachment(s), if any
            for item in items:
                try:
                    children = await self.get_item_children(item.key)
                    pdf_attachment_keys = [
                        child.get("key")
                        for child in children
                        if isinstance(child, dict)
                        and child.get("data", {}).get("itemType") == "attachment"
                        and child.get("data", {}).get("contentType") == "application/pdf"
                    ]
                    for attachment_key in pdf_attachment_keys:
                        if not attachment_key:
                            continue
                        fulltext = await self.get_item_fulltext(attachment_key)
                        if fulltext:
                            item.full_text = fulltext
                            break
                except Exception:
                    # Continue if children/full-text fetch fails for this item
                    pass

                all_items.append(item)

            start += limit

            # Add a small delay to be respectful to the API
            await asyncio.sleep(0.1)

        return all_items
    
    def _parse_item(self, data: Dict[str, Any], key: str) -> ZoteroItem:
        """Parse Zotero API item data into our ZoteroItem model."""
        # Safely extract creators list
        creators = data.get("creators", [])
        if not isinstance(creators, list):
            creators = []
        
        # Safely extract tags
        tags_data = data.get("tags", [])
        tags = []
        if isinstance(tags_data, list):
            for tag in tags_data:
                if isinstance(tag, dict) and tag.get("tag"):
                    tags.append(str(tag.get("tag", "")))
        
        # Safely extract collections
        collections = data.get("collections", [])
        if not isinstance(collections, list):
            collections = []
        
        return ZoteroItem(
            key=str(key) if key else "",
            title=str(data.get("title", "")) if data.get("title") else "",
            creators=creators,
            abstract=str(data.get("abstractNote", "")) if data.get("abstractNote") else "",
            date=str(data.get("date", "")) if data.get("date") else "",
            item_type=str(data.get("itemType", "")) if data.get("itemType") else "",
            collections=collections,
            tags=tags,
            url=str(data.get("url", "")) if data.get("url") else "",
            doi=str(data.get("DOI", "")) if data.get("DOI") else ""
        )