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
    - **Layer Assignment:** Instead of storing a duplicate layer reference, the project registry records the mapping from primitive UUID to its assigned layer.
    - **MOP Assignment:** Similarly, the project registry records a mapping from primitive UUID to the set of associated MOPs.
    - **Parent/Child Linking:**  
      - Instead of storing a child list within each primitive, the project maintains a registry mapping primitive UUIDs to their parent UUID (if any).
      - At XML‐output time, the project “decorates” each primitive’s `<Tag>` with the parent’s UUID.
  - **Classification:**  
    - Each primitive may also have a list of groups and a free-text description.
    - At file output, these values are included in a JSON object in the `<Tag>` element.

- **MOP (Machine Operation, Abstract Base Class):**  
  - **Purpose:** Represents a machining operation.
  - **Intrinsic Attributes:**  
    - Machining parameters and a PID source (which may be a group name or a list of primitive UUIDs).
  - **Relationship:**  
    - The project registry records MOP assignments by mapping a MOP to the list of primitives that should be processed.
    - In case an expected MOP is not found, the project may create a disabled dummy part/MOP to ensure consistency.

---

## 3. Central Relationship Management

### 3.1 Project-Level Registries

The `CamBamProject` object manages not only the entity registries but also dedicated registries for relationships:
- **Layer Registry:**  
  - Maps layer IDs to the primitives assigned to that layer.
  - When a primitive is added or its layer is changed, the project updates this registry.
  - During XML output, the project uses this registry to place primitives in the correct `<objects>` container.
- **MOP Registry:**  
  - Maps MOP IDs to the primitives assigned to each machine operation.
  - The project updates this registry when primitives are added or modified.
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
  - When a primitive is created or updated, any MOP assignments are recorded in the project’s MOP registry.
  - The primitive stores only minimal information (e.g. an empty list), and the project is responsible for keeping the association up to date.
- **Linking Primitives:**  
  - Instead of storing the child list on the primitive, the project maintains a parent mapping: a dictionary mapping each primitive UUID to its parent UUID.
  - Helper methods (e.g., `link_parent(primitive, parent)`) update this mapping.
  - When building the XML, the project queries this registry and injects the parent reference into the `<Tag>` JSON.

### 3.3 Transformation Propagation (Conceptual)

- **Centralized Transformation Methods:**  
  - Transformation methods (e.g., translate, rotate) are implemented at the project level.
  - When a transformation is applied to a primitive, the project looks up all primitives that have that primitive as their parent (via the parent/child registry) and recursively applies the transformation.
  - This ensures that the entire linked tree is updated consistently.
- **Decoupled from Primitive Storage:**  
  - Since the relationship data is not stored within the primitive objects themselves, transformations rely on the project registry for linked children.
  
---

## 4. Serialization and Reconstruction

### 4.1 XML Serialization

- **XML Generation:**  
  - The XML writer (in `cambam_writer.py`) queries the project’s registries to build the file.
  - For each layer, the project uses its layer registry to create a `<layer>` element with an `<objects>` container.
  - Primitives are placed in the correct container based on their `layer_id` (as recorded in the registry).
  - Each primitive’s `<Tag>` element is populated with a JSON object that includes:
    - `user_id`
    - `internal_id`
    - `groups`
    - `parent` (retrieved from the parent/child registry)
    - `description`
  - MOP associations are built by looking up the MOP registry and outputting the list of primitive IDs under each MOP element.
  
### 4.2 Pickle Serialization

- **Native Serialization:**  
  - The project can be serialized using pickle.
  - The project’s registries (including relationship mappings) are stored as part of the project object.
  - Custom `__getstate__` and `__setstate__` methods remove transient or non-serializable attributes (such as weak references).
  
### 4.3 Reconstruction (Importing a CamBam File)

- **XML Reader Module:**  
  - When reconstructing a project from a CamBam XML file, the reader:
    - Reads the layer hierarchy, and for each primitive element, extracts the `<Tag>` JSON.
    - Reconstructs the mapping from the XML-assigned integer primitive IDs back to internal UUIDs.
    - Reassembles the layer, part, and MOP associations by parsing the XML structure:
      - Primitives placed in a given layer container are assigned to that layer.
      - Under each MOP element, the list of primitive XML IDs is used to assign primitives to that MOP.
    - The parent link is re-established from the `<Tag>` JSON.
  - This process allows a project to be reconstructed even if it was originally created externally, provided that the XML file adheres to the expected structure.

---

## 5. Transferring Entities Between Projects

- **Copying Linked Trees:**  
  - The project provides methods to copy (or transfer) an entire primitive tree.
  - The process:
    - Copies the target primitive and its linked children using the parent/child registry.
    - Reassigns new UUIDs if needed while preserving the relative linking structure.
    - Updates the project registries for layers and MOPs accordingly.
- **Registry-Based Linking:**  
  - Since all relationships are maintained in the central project registry, transferring entities involves updating only this registry.
  - The output XML will reflect the new relationships through the `<Tag>` elements and correct container placements.

---

## 6. Summary

This specification defines a robust and decoupled architecture for the CamBam CAD/CAM framework based on a central project object that maintains all relationship data in dedicated registries. Key points include:

- **Centralized Relationship Management:**  
  - The project object holds registries for layer assignments, MOP assignments, and parent/child links.  
  - Primitives themselves do not duplicate relationship data; they contain only intrinsic geometry and transformation state.

- **Simplified Transformation Propagation:**  
  - Transformation operations are executed at the project level.  
  - The project queries its parent/child registry to recursively propagate changes through linked primitives.

- **Robust Serialization and Reconstruction:**  
  - XML output is generated by the writer module using the project’s registries to place primitives in the correct layer and MOP containers.  
  - Each primitive’s `<Tag>` element includes a JSON object with user-friendly and linking data.
  - The framework supports reconstructing a project from XML (and optionally from pickle), ensuring that all relationships are correctly reassembled.

- **Inter-Project Transferability:**  
  - Copy/transfer methods operate on the project’s registries, ensuring that entire linked trees of primitives (and their associated layer/MOP assignments) are transferred without duplication.
  - New UUIDs can be assigned while preserving the relative structure, and the final XML output reflects these updated relationships.

This design minimizes duplication by centralizing relationship management within the project object, and it provides a robust framework for serialization, transformation propagation, and entity transfer between projects.
