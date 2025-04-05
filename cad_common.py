# cad_common.py
import uuid
from dataclasses import dataclass, field
import logging
from typing import Tuple, Sequence

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box."""
    min_x: float = float('inf')
    min_y: float = float('inf')
    max_x: float = float('-inf')
    max_y: float = float('-inf')

    @property
    def width(self) -> float:
        return self.max_x - self.min_x if self.is_valid() else 0.0

    @property
    def height(self) -> float:
        return self.max_y - self.min_y if self.is_valid() else 0.0

    @property
    def center(self) -> Tuple[float, float]:
        if self.is_valid():
            return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)
        else:
            # Or raise error? Returning origin might be misleading.
            logger.warning("Calculating center of an invalid BoundingBox.")
            return (0.0, 0.0)

    def is_valid(self) -> bool:
        return self.min_x <= self.max_x and self.min_y <= self.max_y

    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        if not other.is_valid():
            return self
        if not self.is_valid():
            return other
        return BoundingBox(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y)
        )

    @staticmethod
    def from_points(points: Sequence[Tuple[float, float]]) -> 'BoundingBox':
        if not points:
            return BoundingBox()
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return BoundingBox(min_x, min_y, max_x, max_y)

@dataclass
class CamBamEntity:
    """Base class for all identifiable objects in the project."""
    internal_id: uuid.UUID = field(default_factory=uuid.uuid4)
    user_identifier: str = "" # User-friendly name/ID

    def __post_init__(self):
        # Provide a default user identifier if none is given
        if not self.user_identifier:
            self.user_identifier = f"{self.__class__.__name__}_{self.internal_id.hex[:6]}"

    def __hash__(self):
        # Entities are uniquely identified by their internal ID
        return hash(self.internal_id)

    def __eq__(self, other):
        # Entities are equal if their internal IDs are equal
        if not isinstance(other, CamBamEntity):
            return NotImplemented
        return self.internal_id == other.internal_id

# Custom Exception
class CamBamError(Exception):
    """Base exception for CamBam framework errors."""
    pass

class ProjectConfigurationError(CamBamError):
    """Error related to project setup or entity relationships."""
    pass

class TransformationError(CamBamError):
    """Error during geometric transformations."""
    pass

class XmlParsingError(CamBamError):
    """Error parsing a CamBam XML file."""
    pass

class XmlWritingError(CamBamError):
    """Error writing a CamBam XML file."""
    pass