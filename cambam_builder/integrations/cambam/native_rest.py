"""Editable native rest authoring and explicit original-design certification."""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from ...cam_core import replay, rest_boundaries
from ...native.cad import Pline
from ...native.cam import PocketMop
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from .native_series import NativeSeries, _source_primitives


@dataclass(frozen=True)
class RestBinding:
    """Recomputable evidence; editing a boundary requires fresh derivation/post."""

    source_path: object
    prior_candidate_path: object
    prior_post_path: object
    prior_series: NativeSeries
    boundaries: rest_boundaries.RestBoundaries
    cutting_lengths: tuple
    entry_modes: tuple
    setup: object = None
    cleanup_name: str = "REST"

    @property
    def names(self):
        return tuple(f"{self.cleanup_name}-boundary-{i + 1:02d}"
                     for i in range(len(self.boundaries.windows)))

    @property
    def provenance(self):
        return json.dumps({"format": "native-rest-v1",
                           "original": self.boundaries.target.name,
                           "source": self.prior_series.source_semantic_sha256,
                           "predecessor": self.prior_series.evidence_fingerprint,
                           "derivation": self.boundaries.fingerprint},
                          sort_keys=True, separators=(",", ":"))

    @property
    def predecessor_footer(self):
        height = self.prior_series.items[-1].position[2]
        if height <= 0:
            raise ValueError("native predecessor must return above stock before its spindle stop")
        return f"G0 Z{float(height)}\nM5"

    def check_derivation(self):
        from .native_series_audit import _bind_source_target

        prior = self.prior_series
        prior.check_freshness(self.source_path, self.prior_candidate_path,
                              self.prior_post_path, setup=self.setup)
        target = self.boundaries.target
        _bind_source_target(self.source_path, self.prior_candidate_path, prior, target)
        trace = prior.to_trace({stage.name: target for stage in prior.stages},
                               dict(self.cutting_lengths), dict(self.entry_modes))
        self.predecessor_footer
        project = read_cambam_bytes(Path(self.prior_candidate_path).read_bytes())
        for mop in project.list_mops():
            if mop.enabled and (mop.custom_mop_header or mop.custom_mop_footer not in (
                    "", "M5", f"G0 Z{float(mop.clearance_plane)}\nM5")):
                raise ValueError("native rest predecessor contains unsupported literal transport")
        current = rest_boundaries.derive(
            trace, radius_mm=self.boundaries.radius_mm,
            overlap_mm=self.boundaries.overlap_mm,
            margin_mm=self.boundaries.margin_mm)
        if current != self.boundaries:
            raise ValueError("native rest derivation changed from bound predecessor")
        return trace

    def check_candidate(self, candidate_path):
        """Inspect strict reopened geometry, provenance, selection and MOP intent."""
        project = read_cambam_bytes(Path(candidate_path).read_bytes())
        source = read_cambam_bytes(Path(self.source_path).read_bytes())
        original = _source_primitives(source)
        if _source_primitives(project, original) != original:
            raise ValueError("native rest candidate changed original CAD")
        mops = [m for m in project.list_mops() if m.enabled]
        expected = tuple(stage.name for stage in self.prior_series.stages)
        if tuple(m.name for m in mops) != expected + (self.cleanup_name,):
            raise ValueError("native rest MOP order differs from predecessor and cleanup")
        prior = read_cambam_bytes(Path(self.prior_candidate_path).read_bytes())
        prior_mops = [m for m in prior.list_mops() if m.enabled]
        if tuple(m.name for m in prior_mops) != expected:
            raise ValueError("native rest predecessor MOP order changed")
        # Posted XYZ/tool numbers alone do not encode the cutter footprint,
        # selected design or intended floor. Keep that original native contract
        # when bypassing the ordinary common-target binding for derived Regions.
        fields = ("tool_profile", "tool_number", "tool_diameter", "target_depth",
                  "stock_surface", "work_plane", "custom_mop_header")
        for index, (mop, previous) in enumerate(zip(mops[:-1], prior_mops)):
            footer = (self.predecessor_footer if index == len(prior_mops) - 1
                      else previous.custom_mop_footer)
            targets = tuple(sorted(project.get_primitive(uid).user_identifier
                                   for uid in project.get_mop_targets(mop)))
            prior_targets = tuple(sorted(prior.get_primitive(uid).user_identifier
                                         for uid in prior.get_mop_targets(previous)))
            if (type(mop) is not type(previous) or
                    any(getattr(mop, field) != getattr(previous, field)
                        for field in fields) or targets != prior_targets or
                    mop.custom_mop_footer != footer):
                raise ValueError("native rest predecessor MOP intent changed; regenerate boundaries")
        mop = mops[-1]
        if (type(mop) is not PocketMop or mop.tool_profile != "EndMill" or
                mop.tool_diameter != 2 * self.boundaries.radius_mm or
                mop.tool_number <= 0 or
                f"T{mop.tool_number}" in {s.tool for s in self.prior_series.stages} or
                mop.target_depth != -self.boundaries.target.depth or
                mop.roughing_clearance != 0 or mop.stock_surface != 0 or
                mop.work_plane != "XY" or mop.custom_mop_header or
                mop.custom_mop_footer):
            raise ValueError("native rest requires a fully native cylindrical Pocket at original floor")
        if tuple(sorted(project.get_primitive(uid).user_identifier
                        for uid in project.get_mop_targets(mop))) != tuple(sorted(self.names)):
            raise ValueError("native rest Pocket selections differ from certified boundaries")
        for name, boundary in zip(self.names, self.boundaries.windows):
            primitive = project.get_primitive(name)
            if type(primitive) is not Region or primitive.description != self.provenance:
                raise ValueError("native rest boundary provenance differs")
            coordinates = primitive.get_absolute_coordinates_xyz()
            expected_rings = (boundary.shell,) + boundary.holes
            actual_rings = (coordinates["outer_curve"],) + tuple(coordinates["hole_curves"])
            if (len(expected_rings) != len(actual_rings) or
                    any(tuple((x, y, 0.0, 0.0) for x, y in expected_ring) != tuple(actual)
                        for expected_ring, actual in zip(expected_rings, actual_rings))):
                raise ValueError("native rest boundary geometry differs from derivation")
        return project

    def check(self, source_path, candidate_path, series, target):
        if (Path(source_path).resolve() != Path(self.source_path).resolve() or
                target != self.boundaries.target or type(series) is not NativeSeries):
            raise ValueError("native rest original-design binding differs")
        self.check_derivation()
        self.check_candidate(candidate_path)
        count = len(self.prior_series.stages)
        if (len(series.stages) != count + 1 or
                tuple(replace(s, first_line=0) for s in series.stages[:count]) !=
                tuple(replace(s, first_line=0) for s in self.prior_series.stages) or
                series.stages[-1].target_ids != tuple(sorted(self.names))):
            raise ValueError("native rest posted predecessor or cleanup stages differ from certification")
        # Strip only source line numbers. All actual predecessor setup and motion
        # must recur unchanged in the composed post before trusting its stock.
        previous = tuple(replace(item, line=0) for item in self.prior_series.items)
        prefix = tuple(replace(item, line=0) for item in series.items[:len(previous)])
        if prefix != previous:
            raise ValueError("native rest posted predecessor changed; regenerate boundaries")
        return True


def prepare(prior_series, source_path, prior_candidate_path, prior_post_path, *,
            target, cutting_lengths_mm, entry_modes, radius_mm, overlap_mm,
            margin_mm=0.01, setup=None, cleanup_name="REST"):
    """Certify the predecessor, then derive windows independently of CamBam."""
    from .native_series_audit import audit_linear_native_series, _polygon_target

    if (type(prior_series) is not NativeSeries or
            not isinstance(cleanup_name, str) or not cleanup_name or
            cleanup_name in {s.name for s in prior_series.stages}):
        raise ValueError("unique cleanup MOP name required")
    target = _polygon_target(target)
    audit = audit_linear_native_series(
        prior_series, source_path, prior_candidate_path, prior_post_path,
        targets={s.name: target for s in prior_series.stages},
        cutting_lengths_mm=cutting_lengths_mm, entry_modes=entry_modes,
        section_depths_mm=(target.depth,), max_protected_overcut_mm2=1e-6,
        setup=setup)
    boundaries = rest_boundaries.derive(audit.trace, radius_mm=radius_mm,
                                        overlap_mm=overlap_mm, margin_mm=margin_mm)
    return RestBinding(source_path, prior_candidate_path, prior_post_path,
                       prior_series, boundaries,
                       tuple(sorted(cutting_lengths_mm.items())),
                       tuple(sorted(entry_modes.items())), setup, cleanup_name)


def author(binding, candidate_path, *, tool_number, depth_increment_mm,
           clearance_mm, spindle_rpm, plunge_feed_mm_min, cut_feed_mm_min):
    """Append editable Regions and one native Pocket; strict reopen is mandatory."""
    if type(binding) is not RestBinding:
        raise ValueError("native rest binding required")
    destination = Path(candidate_path).resolve()
    if destination in {Path(path).resolve() for path in (
            binding.source_path, binding.prior_candidate_path, binding.prior_post_path)}:
        raise ValueError("native rest output must preserve original and predecessor inputs")
    binding.check_derivation()
    for value in (depth_increment_mm, clearance_mm, spindle_rpm,
                  plunge_feed_mm_min, cut_feed_mm_min):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("explicit positive native machining controls required")
    if (type(tool_number) is not int or tool_number <= 0 or
            f"T{tool_number}" in {s.tool for s in binding.prior_series.stages} or
            depth_increment_mm > binding.boundaries.target.depth):
        raise ValueError("distinct tool number and bounded depth increment required")
    project = read_cambam_bytes(Path(binding.prior_candidate_path).read_bytes())
    # Default delays the predecessor retract until the next MOP's section and
    # omits M5 at M6. Encode only its already-observed final setup state; native
    # cutting paths remain entirely generated by Pocket/Profile/Engrave.
    enabled = [mop for mop in project.list_mops() if mop.enabled]
    enabled[-1].custom_mop_footer = binding.predecessor_footer
    layer = project.add_layer(f"{binding.cleanup_name} derived boundaries")
    regions = []
    for name, boundary in zip(binding.names, binding.boundaries.windows):
        if project.get_primitive(name) is not None:
            raise ValueError("derived boundary name already exists")
        region = project.add_region(
            layer, Pline(vertices=boundary.shell, closed=True),
            [Pline(vertices=hole, closed=True) for hole in boundary.holes],
            identifier=name, description=binding.provenance)
        if region is None:
            raise ValueError("could not author native rest Region")
        regions.append(region)
    mop = project.add_pocket_mop(
        project.list_parts()[0], targets=regions, name=binding.cleanup_name,
        tool_number=tool_number, tool_diameter=2 * binding.boundaries.radius_mm,
        target_depth=-binding.boundaries.target.depth,
        depth_increment=depth_increment_mm, stock_surface=0, roughing_clearance=0,
        clearance_plane=clearance_mm, tool_profile="EndMill", spindle_speed=spindle_rpm,
        plunge_feedrate=plunge_feed_mm_min, cut_feedrate=cut_feed_mm_min,
        lead_in_type="None", max_crossover_distance=0, optimisation_mode="None")
    if mop is None:
        raise ValueError("could not author native rest Pocket")
    project.save(str(candidate_path))
    return binding.check_candidate(candidate_path)


def audit(binding, candidate_path, post_path, *, cutting_length_mm,
          entry_mode, section_depths_mm, max_protected_overcut_mm2,
          min_new_floor_area_mm2):
    """Replay the complete actual post and require measured useful cleanup.

    Caller-declared virgin entry is stock cutting; cleared descent remains a
    separately checked claim. Native planner provenance needs actual CamBam
    observation; a synthetic post can only test this offline verifier.
    """
    from .native_series import normalize_native_series
    from .native_series_audit import audit_linear_native_series

    if (type(binding) is not RestBinding or
            type(min_new_floor_area_mm2) not in (int, float) or
            not math.isfinite(min_new_floor_area_mm2) or min_new_floor_area_mm2 <= 0):
        raise ValueError("native rest binding and positive useful-removal requirement needed")
    series = normalize_native_series(
        binding.source_path, candidate_path, post_path,
        initial_position=binding.prior_series.initial_position, setup=binding.setup)
    target = binding.boundaries.target
    lengths = dict(binding.cutting_lengths)
    lengths[series.stages[-1].tool] = cutting_length_mm
    modes = dict(binding.entry_modes)
    modes[binding.cleanup_name] = entry_mode
    result = audit_linear_native_series(
        series, binding.source_path, candidate_path, post_path,
        targets={stage.name: target for stage in series.stages},
        cutting_lengths_mm=lengths, entry_modes=modes,
        section_depths_mm=section_depths_mm,
        max_protected_overcut_mm2=max_protected_overcut_mm2,
        setup=binding.setup, derived_binding=binding)
    before, after = result.stages[-2].sections[-1], result.stages[-1].sections[-1]
    if before.remaining_area_mm2[0] - after.remaining_area_mm2[1] < min_new_floor_area_mm2:
        raise ValueError("native rest post lacks required useful floor removal")
    return series, result
