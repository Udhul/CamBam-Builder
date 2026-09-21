"""
CamBam-Builder Framework

This framework provides tools for creating and manipulating CamBam files.

Main entry point:
- CBProject: New WIP implementation with expanded functionality and modularity.
"""

# Import the main project class from the cambam_project module
from .cambam_project import CamBamProject
from .entity_core import Vertex
from .machining_calculations import (
    ActiveConstraint,
    MachineLimits,
    MachiningConstraintError,
    MillingConstraints,
    MillingSolution,
    chip_load_from_feed,
    cutting_power,
    cutting_torque,
    feed_from_chip_load,
    material_removal_rate,
    rpm_from_surface_speed,
    solve_milling_constraints,
    surface_speed_from_rpm,
)
from .machining_recommendations import (
    ApplicableRange,
    CallableRecommendationStrategy,
    DiameterRecommendationTable,
    FixedRecommendationStrategy,
    MachineCapabilities,
    MaterialProfile,
    OperatingConstraints,
    ProfileRecommendationStrategy,
    Recommendation,
    RecommendationContext,
    RecommendationError,
    RecommendationProvenance,
    RecommendationResult,
    RecommendationStrategy,
    StrategyResult,
    ToolProfile,
    recommend_milling,
)
from .machining_planning import (
    DepthPassPlan,
    MillingPlan,
    PlanningDiagnostic,
    plan_depth_passes,
    plan_milling,
)

# Create a shorter alias for the main project class
CBProject = CamBamProject

# Current package version
__version__ = "0.1.0"
