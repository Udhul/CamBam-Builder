"""Generate synthetic A/B files for manual CamBam parent-transform acceptance.

Run from the repository root: python -m demos.generate_parent_validation
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import translation_matrix
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


EXPECTED = {
    "Root": [(20, 10), (20, 20)],
    "Child": [(20, 25), (15, 25)],
    "Grandchild": [(10, 25), (10, 30)],
}


def generate():
    output = Path("output")
    output.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="parent-validation-", dir=output))
    reference = CBProject("A - explicit world coordinates")
    hierarchy = CBProject("B - parent transforms after rejected cycles")
    primitives = {}
    for name, local, color in (
        ("Root", [(0, 0), (10, 0)], "Red"),
        ("Child", [(0, 0), (0, 5)], "Green"),
        ("Grandchild", [(0, 0), (5, 0)], "Blue"),
    ):
        reference.add_pline(reference.add_layer(name + "Layer", color=color), EXPECTED[name], identifier=name)
        primitives[name] = hierarchy.add_pline(
            hierarchy.add_layer(name + "Layer", color=color), local, identifier=name,
        )
    root, child, grandchild = (primitives[name] for name in EXPECTED)
    # Exact 90-degree rotation plus translation, avoiding trig rounding.
    root.effective_transform = np.array([[0., -1., 20.], [1., 0., 10.], [0., 0., 1.]])
    child.effective_transform = translation_matrix(15, 0)
    grandchild.effective_transform = translation_matrix(0, 10)
    assert hierarchy.link_primitive_parent(child, root)
    assert hierarchy.link_primitive_parent(grandchild, child)
    edges = dict(hierarchy._primitive_parent_link)
    assert not hierarchy.link_primitive_parent(root, grandchild)
    assert not hierarchy.link_primitive_parent(child, grandchild)
    assert hierarchy._primitive_parent_link == edges

    a_path = directory / "A_reference.cb"
    b_path = directory / "B_parent_roundtrip.cb"
    save_cambam_file(reference, str(a_path))
    for index in range(2):
        intermediate = directory / f"internal_roundtrip_{index + 1}.cb"
        save_cambam_file(hierarchy, str(intermediate))
        hierarchy = read_cambam_file(str(intermediate))
        assert hierarchy is not None
    save_cambam_file(hierarchy, str(b_path))

    for path in (a_path, b_path):
        tree = ET.parse(path)
        assert len(tree.findall("./layers/layer/objects/*")) == 3
        assert len(tree.findall("./layers/layer")) == 3
        loaded = read_cambam_file(str(path))
        assert loaded is not None
        for name, points in EXPECTED.items():
            np.testing.assert_allclose(
                np.asarray(loaded.get_primitive(name).get_absolute_coordinates())[:, :2],
                points, rtol=0, atol=1e-8,
            )
        assert len(loaded._primitive_parent_link) == (0 if path == a_path else 2)
    print(f"Verified A/B files: {directory.resolve()}")
    return directory


if __name__ == "__main__":
    generate()

