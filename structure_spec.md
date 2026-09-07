# CamBam CAD/CAM Framework – Core Project Structure and Relationship Management Specification

Section 0 describes the implemented architecture; sections 1–7 describe the intended
design, not a verified inventory of implemented behavior. See [current status](PROGRESS.md) for implementation gaps and
the [topic map](docs/README.md) for documentation ownership. MOP PID-source semantics
and central registry ownership require clarification before migration.

This specification describes the core architecture for the CamBam CAD/CAM framework. In this design, all relationships between entities (primitives, layers, parts, and machine operations (MOPs)) are maintained in a central registry managed by the project object. This approach minimizes duplication of relationship data in the individual entities and provides a single source of truth for linking. It also simplifies propagation of transformations, transferring of entities between projects, and robust XML serialization.

---

## 0. Implemented architecture and change ownership

This is a local Python library, with no declared service, database, frontend or
modern command-line entry point. The public entry point is `CamBamProject`, also
exported as `CBProject`. Package declarations include the modern and legacy
packages; `inactive/` and demos are outside that runtime package list.

| Owner | Implemented responsibility | Start here when changing |
| --- | --- | --- |
| `cambam_builder/cambam_project.py` | UUID entity registries, identifier lookup, ordered layers/parts/MOPs, relationship updates, transform orchestration and persistence | Public creation/query/mutation APIs and relationship invariants |
| `cambam_builder/cambam_entities.py` | Entity dataclasses, primitive geometry/bounds, local effective matrices, parent-composed world transforms and entity XML encoding | Geometry or entity fields; inspect reader/writer callers for I/O changes |
| `cambam_builder/cad_transformations.py` | NumPy matrix construction, composition, decomposition and XML matrix conversion | Numerical conventions; inspect entity and project callers together |
| `cambam_builder/cambam_writer.py` | XML ID assignment and layer/part traversal; delegates individual encoding to entities | Output structure and reference resolution |
| `cambam_builder/cambam_reader.py` | XML parsing, entity reconstruction, ID mapping and deferred parent/MOP linking | Import defaults, malformed data and round-trip reconstruction |
| `cambam_builder/__init__.py` | Public alias and version | Import surface and version metadata |
| `cambam_builder/cad_common.py` | Logging configuration with placeholder type/constant sections | Check callers before treating it as an established shared utility layer |

### Data flow and relationship boundaries

1. Callers create and mutate entities through `CamBamProject`. UUIDs key the
   registries; the identifier registry provides human-readable lookup. The project
   maintains both directions of layer membership and parent/child links, group
   indexes, and part/MOP association and order. Update these together through the
   owning APIs rather than editing individual dictionaries in application code.
2. Primitives hold geometry and an effective matrix, plus a weak project reference
   used to compose ancestor transforms. Groups are also represented on entities;
   MOP targeting currently lives in `Mop.pid_source` and resolves through the
   project. The fully centralized target in later sections is not yet implemented.
3. `save()` / `export()` delegate to `save_cambam_file()` and `build_xml_tree()`.
   Entities encode geometry and metadata; primitives export world matrices. The
   reader constructs entities, then resolves references. See the review for known
   fidelity defects; serialization success alone does not establish equivalence.
4. `save_state()` / `load_state()` use Python pickle, restoring primitive project
   links after loading. This is distinct from XML interchange and is not a promised
   version-stable migration format. Trust requirements live in the runbook.

The reader's `PRIMITIVE_TAG_TO_CLASS` and `MOP_TAG_TO_CLASS` are the executable
supported-tag inventory, not a claim of complete CamBam coverage. Consult those
maps and corresponding entity encoders before adding a type.

### XML parent identity and world-pose contract

Primitive `Tag.parent` stores the parent's internal UUID string. XML `mat` stores
the world matrix, while an in-memory `effective_transform` is local to its parent.
The reader snapshots all imported world matrices before linking. For each resolved
parent edge it solves `parent_world @ child_local = child_world`, using those
snapshots rather than partially reconstructed ancestor chains. This preserves
world pose independently of object/layer order and across repeated round trips,
subject to XML numeric precision. Primitive UUIDs, identifiers and layer membership
are retained; layer UUID persistence is not part of the XML format.

Absent, malformed or unresolved parent references leave the primitive parentless
with its imported world matrix. Existing self-parent rejection likewise leaves
the world pose unchanged. A resolved parent with a singular world matrix makes
the entire import fail: the reader logs the child/parent UUIDs and returns `None`
through its existing error boundary. Even a compatible child world matrix cannot
uniquely recover local coordinates; no pseudoinverse or silent detachment is used.
A singular primitive without children is valid. This contract covers acyclic
hierarchies; cycle hardening and general transform/baking fidelity remain separate.

The legacy package exposes `CamBam` and aliases through its own `__init__.py`;
it is a separate implementation, not the modern reader's fallback. Its CLI file
is not a declared installed entry point. Legacy compatibility requires its own
scope and checks. File/artifact handling belongs in the [topic map](docs/README.md).

## 1. Overview

The framework is centered on a **Project Manager** (the `CamBamProject` class) that:
- Maintains registries for each entity type (layers, parts, primitives, and MOPs).
- Uses a **relationship registry** to track all associations between primitives and their parents, as well as assignments to layers and MOPs.
- Provides robust reference resolution so that every entity may be referenced by object, UUID, or user-friendly identifier.
- Enables operations such as adding, removing, copying, and transferring entities while ensuring that all relationships are updated in a single place.
- Supports serialization and reconstruction of a complete project via XML (which includes relationship metadata) and via pickle.

---

## 2. Entity Model

### 2.1 Core Entity Classes

- **CamBamEntity (Base Class):**  
  - Contains an **internal ID** (a UUID) that serves as the primary key.
  - Contains a **user identifier** (a human-friendly string) that must be unique within the project.
  - **Note:** This class does not include relationship information.

- **Layer:**  
  - **Purpose:** Acts as a container for primitives.
  - **Attributes:**  
    - Visual properties (color, alpha, pen width, etc.).
  - **Relationship:**  
    - The project registry maps layer IDs (or unique layer names) to the list of primitives that belong to that layer.

- **Part:**  
  - **Purpose:** Represents a machining part with stock definitions and default machining parameters.
  - **Attributes:**  
    - Stock dimensions, machining origin, and default parameters.
  - **Relationship:**  
    - The project registry associates each part with its corresponding machine operations (MOPs).

- **Primitive (Abstract Base Class):**  
  - **Purpose:** Represents a geometric element (e.g. line, circle, rectangle, text).
  - **Intrinsic Attributes:**  
    - Geometry and transformation state.
  - **Relationship Attributes (Decoupled):**  
    - **Layer Assignment:**  
      - The project registry is solely responsible for mapping each primitive’s UUID to its assigned layer.
    - **MOP Assignment:**  
      - The project registry records the association between primitives and MOPs.  
      - Primitives do not store any MOP assignment data.
    - **Parent/Child Linking:**  
      - The project maintains a registry mapping primitive UUIDs to their parent UUID (if any).
      - At XML‐output time, the project “decorates” each primitive’s `<Tag>` with the parent’s UUID.
  - **Classification:**  
    - Each primitive may also have a list of groups and a free-text description.
    - At file output, these values are included in a JSON object in the `<Tag>` element.

- **MOP (Machine Operation, Abstract Base Class):**  
  - **Purpose:** Represents a machining operation.
  - **Intrinsic Attributes:**  
    - Machining parameters and a PID source (which may be a group name or a list of primitive UUIDs).
  - **Relationship:**  
    - The project registry records MOP assignments by mapping a MOP to the primitives that should be processed.
    - If a referenced MOP is not found, the project may create a disabled dummy MOP (and dummy part) to ensure consistency.

---

## 3. Central Relationship Management

### 3.1 Project-Level Registries

The `CamBamProject` object manages not only the entity registries but also dedicated registries for relationships:
- **Layer Registry:**  
  - Maps layer IDs to the primitives assigned to that layer.
  - When a primitive is added or its layer assignment is changed, the project updates this registry.
  - During XML output, the project uses this registry to place primitives in the correct `<objects>` container.
- **MOP Registry:**  
  - Maps MOP IDs to the primitives assigned to each machine operation.
  - The project is solely responsible for maintaining MOP associations; primitives do not store any MOP assignment data.
  - If a primitive refers to a MOP that does not exist, the project creates a dummy (disabled) MOP.
- **Parent/Child Registry:**  
  - Maintains a mapping from a primitive’s UUID to its parent’s UUID.
  - No duplication occurs inside primitives; the relationship is stored solely in the project.
  - When a primitive is created with a parent, the project records the association in this registry.
  - This registry is used for propagating transformation changes and for serializing the parent–child relationship in the XML `<Tag>` element.

### 3.2 Helper Methods for Relationship Updates

The project provides helper methods to add, remove, or update relationships:
- **Assigning a Layer:**  
  - When a primitive is created (or updated), its intended layer (passed as a unique layer name or object) is registered in the project.
  - The project updates its layer registry so that the primitive is associated with the correct layer.
- **Assigning a MOP:**  
  - When a primitive is created or updated, the MOP association is recorded solely in the project’s MOP registry.
  - The primitive does not store any MOP assignment data; instead, the project injects this information into the `<Tag>` JSON when building the XML.
- **Linking Primitives:**  
  - The project maintains a parent mapping (a dictionary mapping each primitive UUID to its parent UUID).
  - Helper methods update this registry when a primitive is linked to a parent.
  - When building the XML, the project queries this registry and injects the parent reference into the `<Tag>` JSON.

### 3.3 Transformation Propagation (Conceptual)

- **Centralized Transformation Methods:**  
  - Transformation methods (e.g., translate, rotate) are implemented at the project level.
  - When a transformation is applied to a primitive (by updating its transformation matrix or by baking the changes into its geometry), the project looks up any linked children (via the parent/child registry) and recursively applies the transformation.
- **Final State in Primitives:**  
  - The resulting transformation matrix or updated geometry is stored directly in the primitive objects.
- **Decoupled from Relationship Storage:**  
  - Since relationship data is maintained solely in the project’s registries, transformation propagation relies on these registries as the single source of truth.

---

## 4. Serialization and Reconstruction

### 4.1 XML Serialization

- **XML Generation:**  
  - The XML writer (in `cambam_writer.py`) queries the project’s registries to build the final CamBam XML file.
  - For each layer, the project uses its layer registry to create a `<layer>` element with an `<objects>` container.
  - Primitives are placed in the correct container based on the layer assignment provided by the project registry.
  - Each primitive’s `<Tag>` element is populated with a JSON object that includes:
    - `user_id`
    - `internal_id`
    - `groups`
    - `parent` (obtained from the parent/child registry)
    - `description`
- **MOP Associations:**  
  - The XML writer uses the MOP registry to generate MOP sections. Each MOP element includes a `<primitive>` element listing the XML IDs of the primitives associated with that MOP.

### 4.2 Pickle Serialization

- **Native Serialization:**  
  - The project object (including its relationship registries) is serialized using pickle.
  - Custom `__getstate__` and `__setstate__` methods remove transient or non-serializable attributes (such as weak references).

### 4.3 Reconstruction (Importing a CamBam File)

The reconstruction process (in the XML reader module) follows a precise order:
1. **File Base:**  
   - Begin with the root of the XML file.
2. **Parts:**  
   - Parse the `<parts>` element to reconstruct all parts.
3. **MOPs:**  
   - For each part, parse the `<machineops>` element and reconstruct MOPs.
   - Extract the list of primitive XML IDs from each MOP’s `<primitive>` element.
4. **Layers:**  
   - Parse the `<layers>` element to rebuild layer objects.
   - Initially, layers are created without their contained primitives.
5. **Primitives:**  
   - Iterate over each layer’s `<objects>` container to extract primitive elements.
   - For each primitive, parse the `<Tag>` JSON to recover `user_id`, `internal_id`, `groups`, `parent`, and `description`.
   - If the `<Tag>` is missing or incomplete, default values (or new UUIDs) are assigned.
6. **Linking Registries:**  
   - Using the XML structure, the project rebuilds its relationship registries:
     - **Layer Registry:** Each primitive is associated with the layer in which it was found.
     - **Parent/Child Registry:** The parent link is re-established from the `parent` field in the `<Tag>` JSON.
     - **MOP Registry:** For each MOP, the primitive XML IDs (converted back to UUIDs using a temporary mapping) are used to re-associate primitives with the corresponding MOP.
   
This process reverses the file-writing procedure so that a CamBamProject object is fully reconstructed from the XML file, even for files not originally created by the framework (provided the expected metadata is available).

---

## 5. Transferring Entities Between Projects

- **Copying Linked Trees:**  
  - The project provides methods to copy (or transfer) an entire primitive tree.
  - The process uses the project’s parent/child registry to identify the linked tree.
  - **UUID Preservation:**  
    - When transferring or copying primitives between projects, the original UUIDs are preserved in order to maintain the existing relationship links.  
    - If a transferred primitive’s UUID already exists in the target project, it is assumed that the primitive is the same and its data is updated accordingly.
  - The layer and MOP associations are updated in the project’s registries accordingly.
- **Registry-Based Linking:**  
  - Since relationship data is maintained solely in the project’s registries, transferring entities involves updating these registries without duplicating relationship data.
  - The final XML output will reflect the preserved relationships via the `<Tag>` JSON and correct container placements.

---

## 6. Transformation Propagation (Conceptual)

- **Centralized Transformation Handling:**  
  - Although the detailed transformation methods are not covered in this specification, the design requires that transformation operations are executed at the project level.
  - When a transformation is applied to a primitive, the project queries its parent/child registry and recursively propagates the transformation to all linked children.
- **Final State in Primitives:**  
  - The resulting transformation matrix or the updated geometry is stored directly in the primitive objects.
- **Decoupled from Relationship Storage:**  
  - Because relationship data is maintained exclusively by the project, transformation propagation relies on a single source of truth, ensuring consistency even after transferring or reloading entities.

---

## 7. Summary

This specification defines a robust and decoupled architecture for the CamBam CAD/CAM framework based on a central project object that maintains all relationship data in dedicated registries. Key points include:

- **Centralized Relationship Management:**  
  - The `CamBamProject` object holds dedicated registries for layer assignments, MOP associations, and parent/child links.
  - Primitives do not store layer or MOP assignment data internally; these associations are maintained solely by the project and injected into the XML output via the `<Tag>` element.

- **Simplified Transformation Propagation:**  
  - Transformation functions are implemented at the project level and rely on the centralized parent/child registry to recursively propagate changes.
  - The final transformation state or updated geometry resides directly in the primitive objects.

- **Robust Serialization and Reconstruction:**  
  - XML output is generated by querying the project registries, ensuring that primitives are placed in the correct layer and MOP containers and that parent links are recorded in the `<Tag>` metadata.
  - The reconstruction process follows a strict order—starting with parts, then MOPs, layers, primitives, and finally linking primitives to layers, MOPs, and parent relationships—so that a complete project can be rebuilt accurately.
  - The framework supports reconstructing a project from XML (and optionally from pickle), ensuring that all relationships are correctly reassembled.

- **Inter-Project Transferability:**  
  - Copy and transfer methods operate on the centralized registries, allowing entire linked trees of primitives to be transferred without duplicating relationship data.
  - UUIDs are preserved to maintain existing links; if a conflict occurs in the target project, the primitive is updated accordingly.
  - The relative structure is preserved and reflected in the XML output.

This design minimizes duplication by centralizing all relationship data within the project object’s registries, ensuring that the management, propagation, and serialization of entity relationships remain robust and maintainable.
