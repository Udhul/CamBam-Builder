# CamBam CAD/CAM Framework – Fundamental Project Structure Specification

This specification describes the core architecture for the CamBam CAD/CAM framework. It explains how the project manages its entities, stores relationships, and enables robust serialization and transferring between projects. The focus is on the management of assignments and relationships, while the fine‐grained transformation functions (although supported) are abstracted away.

---

## 1. Overview

The framework is built around a central **Project Manager** (the `CamBamProject` class) which:
- Maintains registries for all entities: layers, parts, primitives, and machine operations (MOPs).
- Provides robust reference resolution so that every entity may be referenced by its object instance, unique UUID, or a user-friendly identifier (user ID).
- Manages relationships between entities such as:
  - **Layer assignment:** Each primitive stores a `layer_id` (a UUID or unique layer name) that determines its placement.
  - **MOP assignment:** Primitives store a list of MOP associations in an attribute (e.g. `mop_ids`) using UUIDs.
  - **Parent/Child Linking:** Primitives can be linked together; a primitive may have a parent (stored as `parent_primitive_id`) and a list of child IDs is maintained. This supports hierarchical relationships and transformation propagation.
- Supports transferring or copying entire linked trees of primitives between projects.
- Provides serialization through both a native pickled object and an XML format that is compatible with external CamBam files.

---

## 2. Entity Model and Relationship Management

### 2.1 Core Entity Types

- **CamBamEntity (Base Class):**  
  - Every entity has:
    - An **internal ID** (a UUID) used as the primary key.
    - A **user identifier** (a human-friendly string) which must be unique within the project.
  
- **Layer:**  
  - **Purpose:** Acts as a drawing canvas; primitives are ultimately placed on layers.
  - **Assignment:**  
    - A primitive’s layer is determined solely by its `layer_id`.  
    - Layers themselves are registered in the project with unique names.
  - **Runtime Behavior:**  
    - During XML file generation, the project iterates over the unique layer assignments and builds a container for each layer.
    - If a primitive references a layer that is not found (by its UUID or name), the project automatically creates a new layer using that name.

- **Part:**  
  - **Purpose:** Represents a machining part with stock definitions and default machining parameters.
  - **Assignment:**  
    - A part maintains a list of MOP associations.
  
- **Primitive (Abstract Base Class):**  
  - **Purpose:** Represents a geometric element (line, circle, rectangle, arc, text, etc.).
  - **Assignment Attributes:**  
    - `layer_id`: Stores the UUID (or unique name) of the layer where the primitive belongs.
    - `mop_ids`: A list of UUIDs that indicate which machine operations (MOPs) should process this primitive.
    - `groups`: A list of group names used for organizational or filtering purposes.
    - `description`: A free-text description.
    - `parent_primitive_id`: A UUID linking to a parent primitive (if any).
  - **Linking and Propagation:**  
    - When a primitive is created with a parent, the project updates both the primitive’s `parent_primitive_id` and adds its UUID to the parent’s child list.
    - **Transformation Propagation:**  
      - While transformation methods may be called at the project level, when a transformation is applied to a parent, the system recursively updates all linked children based on their stored relationships.
  - **Serialization:**  
    - When generating the XML file, each primitive writes a `<Tag>` element. This element contains a JSON object with:
      - `user_id`: The user-friendly identifier.
      - `internal_id`: The entity’s UUID.
      - `groups`: The groups to which it belongs.
      - `parent`: The parent primitive’s UUID (or null).
      - `description`: The free-text description.
    - The layer association is maintained via the placement of the primitive in the layer container rather than within the tag.

- **MOP (Machine Operation, Abstract Base Class):**  
  - **Purpose:** Represents machining operations that reference primitives.
  - **Assignment:**  
    - A MOP does not directly store a list of primitives; instead, primitives reference the MOPs they belong to via their `mop_ids`.
    - At file build time, the project reconciles which primitives are associated with which MOPs based on groups and direct references.
  - **Fallback Mechanism:**  
    - If a MOP referenced by a primitive is not found, the system can create a disabled dummy part and a dummy MOP so that the error is visible in the output.

---

## 3. Project Management Structure

### 3.1 CamBamProject Object

The `CamBamProject` class serves as the central hub for the framework. It is responsible for:
- **Registries:**  
  - Maintaining dictionaries (registries) for layers, parts, primitives, and MOPs, keyed by their UUIDs.
  - An identifier registry mapping user-friendly names (IDs) to UUIDs.
  - A grouping registry that maps group names to sets of primitive UUIDs.
- **Entity Relationships:**  
  - Keeping track of which primitives belong to which layers.  
    - When a primitive is created or its layer assignment is updated via project methods, the project updates the primitive’s `layer_id` and the corresponding layer registry.
  - Maintaining parent/child relationships for primitives.
    - Helper methods update both the source object (by setting `parent_primitive_id` and updating the child list) and the project registry.
  - Managing MOP assignments.
    - Primitives store a list of MOP UUIDs. The project holds a registry of MOPs and, when building the XML, uses the project-level associations to assign primitives to MOPs.
    - If a referenced MOP is not found, a dummy (disabled) part and MOP are created.
- **Transformation Context:**  
  - A stack of transformation matrices is maintained at the project level.
  - When a transformation is applied to a primitive (via a project method), the project consults the registry for any linked child primitives and propagates the change recursively.
- **Entity Creation and Lookup:**  
  - All creation methods (e.g., `add_layer`, `add_part`, `add_rect`, etc.) ensure that entities are registered with correct assignments.
  - The project provides a resolver that accepts an entity reference as an object, UUID, or user-friendly identifier.
- **Copying and Transferring:**  
  - The project supports copying an entire primitive tree (a primitive plus all its linked children) and transferring the tree to another project while preserving relative relationships.
  
### 3.2 Decoupled Assignment Updates

Relationship updates are managed by helper methods that update both the object and the project registries:
- **Layer Assignment:**  
  - When a primitive is created or its layer is changed, the project updates the primitive’s `layer_id` and, if needed, registers the primitive under the corresponding layer in the project registry.
  - At file build time, the project iterates over all unique layers and assigns primitives based on their `layer_id`.
- **MOP Assignment:**  
  - Primitives store MOP assignment in the `mop_ids` attribute.
  - The project maintains a registry of MOPs and, during file generation, places primitive references (as integer XML IDs) under the appropriate MOP element.
  - Helper methods allow adding or removing MOP associations for a primitive.
- **Parent/Child Links:**  
  - When a primitive is created with a parent reference, the project records the link by storing the parent's UUID in `parent_primitive_id` and adding the child's UUID to the parent's child list.
  - Transformation propagation and copying methods use these links to recursively affect all children.

---

## 4. Serialization and File Building

### 4.1 XML Serialization

The system supports generating a complete CamBam XML file from a project:
- **Layer Containers:**  
  - The XML writer module iterates over the layer registry. Each layer produces an XML container (e.g., a `<layer>` element with an `<objects>` subelement).
  - Primitives are placed into these containers by matching their stored `layer_id` (which is a UUID or unique layer name).
  - If a primitive’s layer is not found in the registry, the project creates a new layer (using the primitive’s stored layer name) to ensure that every primitive is output.
- **Primitive Tagging:**  
  - Each primitive outputs a `<Tag>` element that stores a JSON object with keys:
    - `user_id`
    - `internal_id`
    - `groups`
    - `parent`
    - `description`
- **Part and MOP Structure:**  
  - Parts are output in a `<parts>` element.
  - Each part contains a `<machineops>` container in which MOPs are listed.
  - Under each MOP element, there is a `<primitive>` element listing the XML IDs (as integers) of the primitives associated with that MOP.
- **Reconstruction:**  
  - The reader module (or a separate import process) is designed to reconstruct a CamBamProject object by:
    - Iterating over layers to extract primitives (and reading the `<Tag>` JSON to recover user_id, internal_id, groups, parent, description).
    - Iterating over parts and MOPs to rebuild MOP associations.
  - This process reverses the file‐writing procedure so that a project can be rebuilt even from a CamBam file not produced by this framework.

### 4.2 Pickle Serialization

- **Native Serialization:**  
  - The project can also be serialized using Python’s pickle mechanism.
  - Custom state methods ensure that non-serializable attributes (e.g. weak references) are removed before pickling.
  
### 4.3 Transferring Entities Between Projects

- **Linked Primitive Transfer:**  
  - The framework supports copying (or transferring) an entire tree of linked primitives.
  - The project-level transfer method reassigns new UUIDs as needed while preserving parent/child relationships.
  - All associations (layer, MOP, groups) are updated in the registries.
  
---

## 5. Transformation Propagation Consideration

While the detailed transformation methods are not the focus of this specification, the design requires that:
- Transformation operations are ideally executed at the project level.
- When a transformation is applied to a parent primitive, the project consults the registry for any linked children (using the stored parent/child links) and recursively applies the transformation.
- This design decouples transformation propagation from the primitive objects themselves and ensures consistency when entities are transferred or reloaded.

---

## 6. Summary

This specification describes a fundamental project structure for the CamBam CAD/CAM framework that:

- **Decouples Assignment from Transformation:**  
  The project manages relationships (layer, MOP, parent/child) via a central registry and helper methods. Primitives store their own assignment attributes (e.g. `layer_id`, `mop_ids`, `groups`, `description`, and `parent_primitive_id`), while the project is responsible for maintaining consistent registries and updating these relationships when changes occur.

- **Robust Serialization:**  
  The XML writer module builds a complete CamBam file by placing primitives in layer containers and MOP sections, while encoding structured metadata (via a JSON object in `<Tag>`) to capture user-friendly identifiers, groups, parent links, and descriptions. A reader module can reconstruct a project from this XML file. In addition, native pickle serialization is supported with custom state management.

- **Inter-Project Transferability:**  
  The project-level transfer and copy methods allow entire linked trees of primitives (with their hierarchical relationships) to be moved between projects, with all associations updated accordingly.

- **Transformation Propagation:**  
  Although transformation details are deferred, the design ensures that any transformation applied to a parent primitive is recursively propagated to its linked children by consulting the project’s relationship registries.

This architecture forms a robust, decoupled foundation for the CamBam framework, allowing future extension with detailed transformation operations and additional CAD helper functions while ensuring that entity relationships and serialization remain consistent and transparent.
