# CamBam Builder Progress Tracker

This document outlines the development plan required to complete the framework according to `structure_spec.md`.  The scope only covers the modern package in `./cambam_builder`.

## Current state (v0.1.0)
* Dataclass based entity model (`Layer`, `Part`, `Primitive` subclasses and several `Mop` types).
* `CamBamProject` keeps registries for entities and records layer membership, parent/child links, group membership and part→MOP ordering.
* XML writer/reader exist and support round‑tripping projects.
* Basic transformation utilities and project level methods (translate, rotate, scale, baking etc.).
* Pickle based `save_state` / `load_state` for project persistence.
* No central mapping of MOPs to their primitives – MOP objects store `pid_source` internally.
* Missing cross‑project copy/transfer utilities and only limited tests/demonstrations.
* Several TODOs remain (bounding box accuracy, matrix helpers, API streamlining, validation etc.).

## Target
The final release must fully implement the architecture described in `structure_spec.md`: all relationships managed in the project object, robust XML import/export, transformation propagation, and utilities for transferring entities between projects.

Below is a series of product increments to reach that state.

### Increment 1 – Registry alignment and cleanup
- Refactor `Mop` handling so primitive assignments are stored in project registries rather than `pid_source` on the MOP instance.  Provide APIs to set/get a MOP’s primitive list and keep backward compatibility with group names.
- Add internal checks when linking primitives to parents to prevent circular references and invalid identifiers.
- Resolve outstanding TODOs in transformation helpers (matrix conversions) and in `bake_primitive_transform` logic.
- Improve error handling and logging in identifier resolution functions.

### Increment 2 – Serialization consistency
- Update the XML writer to pull MOP→primitive associations from the new registry and ensure every primitive `<Tag>` contains `user_id`, `internal_id`, `groups`, `parent`, and `description` fields.
- Enhance the reader to rebuild the new MOP registry and to load project level defaults when available.
- Guarantee that missing or malformed `<Tag>` data still results in valid entities with generated UUIDs and unique user identifiers.
- Implement pretty‑printing and deterministic ordering for all exported XML to aid version control comparisons.

### Increment 3 – Copy and transfer utilities
- Introduce APIs to copy primitives (optionally including their child trees) within a project.
- Add functions to transfer primitives and their relationships between two `CamBamProject` instances while preserving UUIDs and updating layer/MOP membership in the target project.
- Expose convenience methods for duplicating layers or parts while keeping their internal ordering and links.

### Increment 4 – Transformation propagation and geometry helpers
- Ensure all transformation methods recursively affect child primitives according to the parent/child registry.  Provide unit tests for translate, rotate, scale, mirror and align operations with and without baking.
- Improve bounding box calculations for primitives that use bulges or arcs so that collision and alignment logic are reliable.
- Add optional utilities for calculating overall project bounding boxes and centering / aligning groups of primitives.

### Increment 5 – Testing, documentation and packaging
- Create a pytest based test suite covering entity creation, relationship management, transformation propagation and XML import/export.
- Expand the README with usage examples and document the architectural design summarised from `structure_spec.md`.
- Package metadata in `pyproject.toml`/`setup.py` should be updated for distribution on PyPI and to expose command line helpers if desired.
- Provide example demo scripts in `demos/` using the new APIs.

### Final Release (v1.0)
- All features from the specification implemented and tested.
- Full project documentation and API reference generated.
- Stable backwards compatible loader kept in `legacy_cambam_builder` for reading older project files.
