"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Example schemas (keep for reference)

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# AI YouTube Clipper Schemas

class ClipSuggestion(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcript text inside the clip window")
    score: float = Field(..., description="Relevance score for ranking")
    title: str = Field(..., description="Short human-friendly title for the clip")

class ClipRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    max_clips: int = Field(5, ge=1, le=10, description="Maximum number of clips to return")
    clip_length: int = Field(30, ge=10, le=120, description="Length of each clip in seconds")
    language: Optional[str] = Field(None, description="Preferred transcript language (e.g., 'en')")

class ClipAnalysis(BaseModel):
    url: str
    video_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    thumbnail: Optional[str] = None
    suggestions: List[ClipSuggestion]
    summary: Optional[str] = None
    language: Optional[str] = None
    clip_length: int
    max_clips: int
    raw_stats: Optional[Dict[str, Any]] = None
