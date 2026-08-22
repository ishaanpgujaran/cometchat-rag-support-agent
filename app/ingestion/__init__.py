"""app/ingestion — knowledge-base parsing and chunking."""

from app.ingestion.models import Chunk, ChunkMetadata
from app.ingestion.parser import parse_directory, parse_file

__all__ = ["Chunk", "ChunkMetadata", "parse_file", "parse_directory"]
