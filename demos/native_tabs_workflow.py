"""Prepare paired native tab documents, then audit their actual Default posts.

Run from the repository root with the declared interpreter and planar extra.
Preparation writes editable CamBam sources only. Generate the six corresponding
``.nc`` files in CamBam, then run this consumer again with ``--audit``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from cambam_builder.integrations.cambam.native_tabs import (
    NativeTabDocument, audit_post, create_document,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


LAYOUTS = {
    "centered": ((40, 10), (70, 25), (40, 40), (10, 25)),
    "moved": ((40, 10), (70, 25), (40, 40), (10, 33)),
    "added": ((40, 10), (70, 25), (40, 40), (10, 33), (10, 17)),
}


def prepare(directory: Path) -> dict:
    """Create six sources and prove library reopen and ordinary tab editing."""
    directory.mkdir(parents=True, exist_ok=False)
    cases = {}
    for style in ("Square", "Triangle"):
        for layout, points in LAYOUTS.items():
            name = f"{style.lower()}-{layout}"
            document = create_document(style=style, points=points)
            source = directory / f"{name}.cb"
            source.write_bytes(document.xml_bytes)
            reopened = read_cambam_bytes(document.xml_bytes, strict=True)
            document.check_document(serialize_cambam_bytes(reopened))
            # Native callers can move/add tabs on a reopened MOP. Regenerate
            # and bind the edited document before interpreting its actual post.
            mop = reopened.get_mop("manual-tab-fixture")
            mop.tab_min_tabs = mop.tab_max_tabs = len(LAYOUTS["added"])
            mop.manual_tab_points = list(LAYOUTS["added"])
            edited = serialize_cambam_bytes(reopened)
            edited_project = read_cambam_bytes(edited, strict=True)
            observed = edited_project.get_mop("manual-tab-fixture").inspected_manual_tab_points(
                edited_project)
            if observed is None or sorted((round(x, 6), round(y, 6)) for x, y in observed) != sorted(LAYOUTS["added"]):
                raise ValueError("tab move/add edit did not survive native round trip")
            cases[name] = {"style": style, "points": points,
                           "source_sha256": hashlib.sha256(document.xml_bytes).hexdigest(),
                           "document": document.check_document(),
                           "edit_roundtrip": "pass"}
    manifest = {"format": "native-tabs-consumer-v1", "cases": cases,
                "actual_posts": "not_evaluated"}
    (directory / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


def audit(directory: Path) -> dict:
    """Re-read all six exact sources/posts; reject changed prepared sources."""
    manifest = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    expected_names = {f"{style}-{layout}" for style in ("square", "triangle")
                      for layout in LAYOUTS}
    if (manifest.get("format") != "native-tabs-consumer-v1"
            or set(manifest.get("cases", {})) != expected_names):
        raise ValueError("complete prepared Square/Triangle comparison required")
    reports = {}
    for name, case in manifest["cases"].items():
        style_name, layout = name.split("-", 1)
        style, points = style_name.title(), LAYOUTS[layout]
        if (case.get("style") != style
                or case.get("points") != [list(point) for point in points]):
            raise ValueError(f"{name}: prepared comparison intent changed")
        source_bytes = (directory / f"{name}.cb").read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != case["source_sha256"]:
            raise ValueError(f"{name}: prepared source changed; prepare fresh evidence")
        document = NativeTabDocument(style, points, source_bytes)
        reports[name] = audit_post(document, (directory / f"{name}.nc").read_bytes())
    report = {"status": "pass", "cases": reports,
              "scope": "native documents, decoded Default motion and retained stock",
              "physical_cutting": "not_evaluated"}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                          encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path("output") / f"native-tabs-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--audit", action="store_true",
                        help="audit actual matching Default .nc files in the prepared directory")
    args = parser.parse_args()
    result = audit(args.directory) if args.audit else prepare(args.directory)
    print(json.dumps(result, indent=2))
