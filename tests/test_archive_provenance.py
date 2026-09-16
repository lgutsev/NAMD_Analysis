import json

from namd_analysis import archive_provenance
from namd_analysis.populations import StateMap


def _write_plain_shprop(path, rows=3):
    lines = []
    for step in range(1, rows + 1):
        # time, energy, six complete populations; deliberately no header.
        lines.append(
            f"{step:.2f} -0.8 0.0 0.0 0.0 0.0 0.0 1.0\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _manifest(tmp_path, cycle_length=3):
    frames = []
    for frame in range(1, cycle_length + 1):
        procar = tmp_path / f"PROCAR.{frame}"
        procar.write_text("placeholder\n", encoding="utf-8")
        frames.append({"frame": frame, "procar": str(procar)})
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"frames": frames, "cycle_length": cycle_length}),
        encoding="utf-8",
    )
    return path


def _state_map(name="FAPI_001_BCF_PCBM_A", band_numbers=None):
    payload = {
        "name": name,
        "time_column": 0,
        "time_unit": "fs",
        "population_columns": [2, 3, 4, 5, 6, 7],
        "groups": {
            "VBM": [2],
            "BCF": [3],
            "PCBM": [4, 5, 6],
            "CBM": [7],
        },
        "complete_population": True,
        "recombined_group": "VBM",
    }
    if band_numbers is not None:
        payload["band_numbers"] = band_numbers
    return StateMap.from_dict(payload)


def test_plain_real_archive_uses_registered_basis_and_filename_namdtini(tmp_path):
    archive_provenance.install()
    shprop = tmp_path / "SHPROP.37"
    _write_plain_shprop(shprop)
    manifest = _manifest(tmp_path)

    plan = archive_provenance.plan_analysis(
        [shprop], _state_map(), manifest, "dish-cyclic"
    )

    assert plan.bands == [976, 977, 978, 979, 980, 981]
    assert plan.alignments[0]["band_numbers_source"] == "preset_bcf_pcbm_A"
    assert plan.alignments[0]["NAMDTINI"] == 37
    assert plan.alignments[0]["NAMDTINI_source"] == "SHPROP_filename_suffix"
    assert plan.alignments[0]["cycle_length_source"] == "projection_manifest"
    assert plan.frames_by_file[0].tolist() == [1, 2, 3]


def test_explicit_state_map_band_numbers_override_campaign_inference(tmp_path):
    archive_provenance.install()
    shprop = tmp_path / "SHPROP.5"
    _write_plain_shprop(shprop)
    manifest = _manifest(tmp_path)
    bands = [100, 101, 102, 103, 104, 105]

    plan = archive_provenance.plan_analysis(
        [shprop], _state_map(name="custom", band_numbers=bands), manifest, "dish-cyclic"
    )

    assert plan.bands == bands
    assert plan.alignments[0]["band_numbers_source"] == "state_map.band_numbers"


def test_filename_and_optional_namdtini_metadata_must_agree(tmp_path):
    archive_provenance.install()
    shprop = tmp_path / "SHPROP.37"
    shprop.write_text(
        "# NAMDTINI = 38\n"
        "1.00 -0.8 0 0 0 0 0 1\n"
        "2.00 -0.8 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    record = archive_provenance.shprop_structure(shprop)

    try:
        archive_provenance.resolve_namdtini(record)
    except ValueError as exc:
        assert "provenance sources disagree" in str(exc)
    else:
        raise AssertionError("conflicting NAMDTINI provenance was accepted")


def test_state_map_roundtrip_keeps_explicit_band_numbers():
    archive_provenance.install()
    bands = [976, 977, 978, 979, 980, 981]
    state_map = _state_map(name="custom", band_numbers=bands)

    assert state_map.band_numbers == bands
    assert state_map.as_dict()["band_numbers"] == bands
