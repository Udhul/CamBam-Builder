# CamBam CAD/CAM Framework – Fundamental Project Structure Specification

This specification outlines the design and core architecture of the CamBam CAD/CAM project management framework. Its primary goals are to provide a robust, decoupled management of entities and their relationships while allowing for transformation propagation through linked entities and seamless serialization between CamBam XML files and project objects.

---

## 1. Overview

The framework is designed to support the creation and manipulation of CamBam projects through a central project class. It manages multiple entity types:
- **Layers:** Represent drawing canvases in which primitives reside.
- **Parts:** Define machining parts, including their stock dimensions and associated machine operations.
- **Primitives:** Basic geometric entities (such as lines, circles, rectangles, arcs, text, etc.) that are drawn on layers.
- **MOPs (Machine Operations):** Represent machining instructions that reference primitives (via groups or direct identifiers).

At the heart of the framework is the **Project Manager (CamBamProject)**, which maintains registries for all entities, provides robust reference resolution, and supports operations such as adding, removing, copying, and transferring entities.

---

## 2. Entity and Relationship Model

### 2.1 Entities

- **CamBamEntity (Base Class):**  
  Every entity (layer, part, primitive, MOP) inherits from a common base that defines:
  - **Internal ID:** A unique UUID that serves as the primary key.
  - **User Identifier:** A human-friendly string identifier.  
  This base class ensures that each entity can be referenced unambiguously throughout the project.

- **Layer:**  
  - **Purpose:** Serves as the container for primitives.  
  - **Attributes:**  
    - `color`, `alpha`, `pen_width`, `visible`, `locked`
    - **Primitive IDs:** A set of UUIDs corresponding to the primitives assigned to this layer.
  - **Linking:**  
    - When a primitive is created and assigned a layer, its `layer_id` stores the UUID of that layer.  
    - The project registers the layer and its ordering so that XML output places primitives into the proper container.

- **Part:**  
  - **Purpose:** Represents a machining part.  
  - **Attributes:**  
    - Stock dimensions, machining origin, and default parameters (such as tool diameter and spindle speed).
    - **MOP IDs:** A list of UUIDs that reference the machine operations associated with the part.

- **Primitive (Abstract Base Class):**  
  - **Purpose:** Represents a drawable geometric entity.  
  - **Key Assignment Attributes:**  
    - `layer_id`: The UUID of the layer on which the primitive is drawn.
    - `groups`: A list of group names for classification.
    - `description`: Free-text description.
    - `mop_ids`: A list of UUIDs referencing the associated MOPs (if any).
    - `parent_primitive_id`: Optional UUID of a parent primitive (used to define hierarchical, linked structures).
  - **Linking:**  
    - **Parent/Child Relationship:**  
      - Each primitive may have a parent.  
      - When a primitive is created with a parent, it stores the parent’s UUID in `parent_primitive_id` and registers its own UUID in the parent’s child list.
    - **Transformation Propagation (Conceptual):**  
      - Although transformation methods are defined on primitives, the design requires that when a transformation is applied to a parent, the change propagates recursively to all linked child primitives.
  - **Serialization:**  
    - When generating XML, each primitive outputs a `<Tag>` element that stores a JSON object with:
      - `user_id`: The user-friendly identifier.
      - `internal_id`: The UUID.
      - `groups`: The list of groups.
      - `parent`: The parent primitive’s UUID (if any).
      - `description`: The free-text description.
    - The layer assignment is implicit via the placement of the primitive XML in the correct layer container.
  
- **MOP (Machine Operation, Abstract Base Class):**  
  - **Purpose:** Represents machining instructions that reference primitives.
  - **Attributes:**  
    - `pid_source`: A reference (either a group name or a list of primitive UUIDs) that indicates which primitives are targeted.
    - Additional machining parameters.
  - **Linking:**  
    - The association between MOPs and primitives is established indirectly: primitives are classified into groups and/or directly referenced via UUIDs. The XML writer uses this information to list the primitives under each MOP.
  
---

## 3. Project Management Structure

### 3.1 Central Project Object

- **CamBamProject:**  
  This object is the central manager and contains:
  - **Registries:**  
    - Dictionaries mapping UUIDs to entities (primitives, layers, parts, MOPs).
    - An identifier registry mapping user-friendly names to UUIDs.
    - A grouping registry for primitives (to support MOP lookup via group names).
  - **Entity Order Lists:**  
    - Lists to preserve the ordering of layers and parts for XML output.
  - **Transformation Context:**  
    - A stack of transformation matrices used to compute the effective transformation for newly created primitives.
  - **Cursor Position:**  
    - A global offset (cursor) used when creating primitives.
  
- **Entity Creation and Lookup:**  
  - **Reference Resolution:**  
    - Methods (e.g., `_resolve_identifier`) allow the project to accept references as an object, a UUID, or a user-friendly string.
  - **Creation Methods:**  
    - Methods such as `add_layer`, `add_part`, and `add_rect` create new entities, register them in the project, and set assignment attributes (e.g., storing the correct `layer_id`).
  - **Linking and Parent/Child Management:**  
    - When a primitive is added with a parent reference, the project updates the primitive’s `parent_primitive_id` and adds the primitive’s UUID to the parent’s child list.
  - **Copy and Transfer:**  
    - Methods allow for recursively copying a primitive along with its entire child tree. This ensures that, when transferring between projects, the complete linked structure is maintained.
  
### 3.2 Decoupling Assignments from Transformation Logic

- **Separation of Concerns:**  
  - The project object manages assignments (layer, part, groups, MOPs) and linking independently from the transformation operations.
  - Transformation methods are defined on the primitive base class and are responsible for updating the effective transformation matrix. They are designed to also trigger updates (or “baking”) on all linked children.
- **Link Propagation:**  
  - The parent/child relationship is maintained as a set of child UUIDs on each primitive. When a transformation is applied to a parent, the propagation method ensures that each child’s transformation is updated consistently.
- **Serialization Considerations:**  
  - During XML output, the linking information is encoded in the `<Tag>` element.  
  - When transferring a primitive between projects, the entire linked tree is processed, reassigning new UUIDs as needed, while preserving the relative linking structure.
  
---

## 4. Serialization and Transfer

### 4.1 XML Serialization

- **XML Structure:**  
  - **Layer Containers:**  
    - Each layer’s XML container is built from the layer registry.  
    - Primitives are placed into the container corresponding to their `layer_id`.
  - **Primitive Tagging:**  
    - Each primitive outputs a `<Tag>` element containing a JSON object with:
      - `user_id`, `internal_id`, `groups`, `parent`, and `description`
    - The association to MOPs is not directly encoded here since primitives are later associated with MOPs based on groups or direct references.
  - **Parts and MOPs:**  
    - Parts are output with their associated machine operations.  
    - MOPs include references to primitives via the previously established identifier or group linkage.

### 4.2 Pickle Serialization

- **Project State Saving:**  
  - The project object can be serialized using Python’s pickle mechanism.
  - Custom `__getstate__` and `__setstate__` methods in primitives and other entities ensure that non-serializable attributes (like weak references) are excluded.
  
### 4.3 Transferring Entities Between Projects

- **Inter-Project Transfer:**  
  - When a primitive (and its linked children) is transferred from one project to another, the entire tree is copied.  
  - The project’s copy/transfer methods reassign new UUIDs where necessary while maintaining relative parent/child relationships.
  - The linking data stored in the XML `<Tag>` element enables reconstructing these relationships when importing a CamBam file.
  
---

## 5. Future Extension: Transformation Propagation

While the details of transformation methods are not covered here, note the following design considerations:

- **Primitive Transformation Methods:**  
  - Each primitive will have methods (e.g., translate, rotate, scale) that update its effective transformation matrix.
- **Propagation to Linked Children:**  
  - When a transformation is applied to a primitive, all primitives linked as children must have their transformation matrices updated accordingly.
  - This propagation must work seamlessly even when the primitive tree is transferred between projects or when reloaded from serialized files.
  
---

## 6. Summary

This specification defines a fundamental project structure for the CamBam framework that:

- Separates management of entity assignments (layers, parts, groups, descriptions, MOP associations) from transformation logic.
- Uses a central project object (CamBamProject) that maintains registries for all entities, an identifier registry for robust reference resolution, and supports linking between primitives (via parent/child relationships).
- Supports XML serialization in which each primitive’s classification (user ID, internal ID, groups, parent, and description) is stored in a structured JSON object within a `<Tag>` element. Layer and MOP associations are derived from the entity’s placement in the XML structure.
- Provides for the transfer and copying of linked entities between projects, ensuring that the complete hierarchical structure is preserved.
- Leaves room for the future integration of transformation propagation across linked primitives.

This design lays a solid foundation for a decoupled, maintainable framework that can later be extended with full transformation and CAD helper functionality.
