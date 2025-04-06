# CamBam CAD/CAM Framework – Core Project Structure and Relationship Management Specification

This specification describes the core architecture for the CamBam CAD/CAM framework. In this design, all relationships between entities (primitives, layers, parts, and machine operations (MOPs)) are maintained in a central registry managed by the project object. This approach minimizes duplication of relationship data in the individual entities and provides a single source of truth for linking. It also simplifies propagation of transformations, transferring of entities between projects, and robust XML serialization.

---

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
      - The project registry is solely responsible for tracking which primitives are processed by which MOPs.
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
    - The project registry records the association between a MOP and the primitives that should be processed.
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
  - No duplication occurs within primitives; relationship data is stored solely in the project.
  - When a primitive is created with a parent, the project records the association in this registry.
  - This registry is used for propagating transformation changes and for serializing the parent–child relationship in the XML `<Tag>` element.

### 3.2 Helper Methods for Relationship Updates

The project provides helper methods to add, remove, or update relationships:
- **Assigning a Layer:**  
  - When a primitive is created (or updated), its intended layer (passed as a unique layer name or object) is registered in the project.
  - The project updates its layer registry so that the primitive is associated with the correct layer.
- **Assigning a MOP:**  
  - When a primitive is created or updated, the project registers the MOP association solely in its MOP registry.
  - No MOP assignment data is stored directly on the primitive; instead, the project injects this information into the `<Tag>` JSON when building the XML.
- **Linking Primitives:**  
  - The project maintains a parent mapping (a dictionary mapping each primitive UUID to its parent UUID, for those primitives that have linked parents).
  - Helper methods update this registry when a primitive is linked to a parent.
  - When building the XML, the project queries this registry and injects the parent reference into the `<Tag>` JSON.

### 3.3 Transformation Propagation (Conceptual)

- **Centralized Transformation Methods:**  
  - Transformation methods (e.g., translate, rotate) are implemented at the project level.
  - When a transformation is applied to a primitive, the project queries its parent/child registry to determine if linked children exist and recursively applies the transformation.
- **Decoupled from Primitive Storage:**  
  - Because relationship data is maintained only in the project’s registries, transformation functions use the registry as the single source of truth for propagating changes.

---

## 4. Serialization and Reconstruction

### 4.1 XML Serialization

- **XML Generation:**  
  - The XML writer (in `cambam_writer.py`) queries the project’s registries to build the final CamBam XML file.
  - For each layer, the project uses the layer registry to create a `<layer>` element with an `<objects>` container.
  - Primitives are placed into the correct container based on their assigned layer (as determined by the project registry).
  - Each primitive’s `<Tag>` element is populated with a JSON object that includes:
    - `user_id`
    - `internal_id`
    - `groups`
    - `parent` (obtained from the parent/child registry)
    - `description`
- **MOP Associations:**  
  - The XML writer uses the MOP registry to generate MOP sections, where each MOP element includes a `<primitive>` element listing the XML IDs of the primitives associated with that MOP.
  
### 4.2 Pickle Serialization

- **Native Serialization:**  
  - The project object (including its relationship registries) can be serialized using pickle.
  - Custom state management (`__getstate__`/`__setstate__`) ensures that non-serializable attributes (e.g., weak references) are omitted.

### 4.3 Reconstruction (Importing a CamBam File)

- **XML Reader Module:**  
  - The reader parses the XML file and reconstructs the project by:
    - Iterating over layer elements to recover primitives and their container assignments.
    - Extracting each primitive’s `<Tag>` JSON to recover `user_id`, `internal_id`, `groups`, `parent`, and `description`.
    - Reassembling the project’s registries:
      - The layer registry is rebuilt from the layer container structure.
      - The parent/child registry is rebuilt using the parent data from each primitive’s `<Tag>`.
      - MOP associations are reconstructed from the primitive IDs listed under each MOP element.
  - This process allows reconstruction of the complete CamBamProject, including all relationships, even from external CamBam files (provided the expected metadata is present).

---

## 5. Transferring Entities Between Projects

- **Copying Linked Trees:**  
  - The project provides methods to copy or transfer an entire tree of linked primitives.
  - The process:
    - Uses the parent/child registry to identify the entire linked tree.
    - Copies the primitives while assigning new UUIDs as needed, preserving the relative structure.
    - Updates the project’s registries (layer, MOP, and parent/child) accordingly.
- **Registry-Based Linking:**  
  - Since relationship data is maintained only in the project’s registries, transferring entities involves updating these registries without duplicating relationship information.
  - The output XML reflects the new relationships via the `<Tag>` JSON and correct container placements.

---

## 6. Transformation Propagation (Conceptual)

- **Centralized Transformation Handling:**  
  - Although the detailed transformation methods are not covered in this specification, the design requires that:
    - Transformation operations are executed at the project level.
    - When a transformation is applied to a primitive, the project queries its parent/child registry and recursively propagates the transformation to all linked children.
- **Decoupled from Relationship Storage:**  
  - Because relationship data is maintained exclusively by the project, transformation propagation relies on a single source of truth, ensuring consistency even after transferring or reloading entities.

---

## 7. Summary

This specification defines a robust, decoupled architecture for the CamBam CAD/CAM framework based on centralized relationship management. Key points include:

- **Centralized Relationship Management:**  
  - The `CamBamProject` object holds dedicated registries for layer assignments, MOP associations, and parent/child links.  
  - Primitives do not store layer or MOP assignment data internally; these associations are maintained solely by the project and injected into the XML output.
- **Simplified Transformation Propagation:**  
  - Transformation functions are implemented at the project level and rely on the centralized parent/child registry to recursively propagate changes.
- **Robust Serialization and Reconstruction:**  
  - XML output is generated by querying the project registries, ensuring that primitives are placed in the correct layer and MOP containers and that parent links are recorded in the `<Tag>` metadata.
  - The XML reader can reconstruct the project by reversing this process, while pickle serialization is supported for native project state storage.
- **Inter-Project Transferability:**  
  - Copy and transfer methods operate on the centralized registries, allowing entire linked trees of primitives to be transferred without duplicating relationship data.
  - New UUIDs are assigned where necessary, and the relative structure is preserved and reflected in the XML output.

This design minimizes duplication by keeping all relationship data within the project object’s registries, ensuring that the management, propagation, and serialization of entity relationships remain robust and maintainable.
