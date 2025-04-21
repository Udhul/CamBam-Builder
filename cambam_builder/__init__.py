"""
CamBam-Builder Framework

This framework provides tools for creating and manipulating CamBam files.

Main entry points:
- CBProject: New WIP implementation with expanded functionality and modularity.
- LegacyCBProject: Legacy implementation with limited flexibility.
"""

# Import the main project class from the cambam_project module
from .cambam_project import CamBamProject

# Create a shorter alias for the main project class
CBProject = CamBamProject

# Import the legacy implementation with a descriptive name
from .legacy.legacy_cambam_builder import CamBam as LegacyCamBamProject

# Create a shorter alias for the legacy project class
LegacyCBProject = LegacyCamBamProject

# Current package version
__version__ = "0.1.0"