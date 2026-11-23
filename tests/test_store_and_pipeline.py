import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from network_generator.oecd_icio import build, load_releases
from network_generator.oecd_icio.archive import file_sha256
from network_generator.oecd_icio.store import TableNotFound, TableStore, package_for_mirror
from tests.test_oecd_icio import RELEASE, write_toy_icio

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def mirror(tmp_path):
    """A toy build packaged as mirror assets, served from a file:// URL."""
    src = tmp_path / "src"
    src.mkdir()
    for year in (2014, 2015):
        write_toy_icio(src / f"ICIO2021_{year}.csv")
    build(RELEASE, [src], tmp_path / "build")
    package_for_mirror(tmp_path / "build", tmp_path / "mirror")
    pin = file_sha256(tmp_path / "mirror" / "MANIFEST.json")
    release = dataclasses.replace(
        RELEASE, mirror_url=(tmp_path / "mirror").as_uri(), mirror_manifest_sha256=pin
    )
    return release, tmp_path


def test_mirror_download_matches_build_and_is_cached(mirror):
    release, tmp = mirror
    store = TableStore(release, cache_root=tmp / "cache")

    table = store.get("CHN", 2015, data_dir=None)

    with (tmp / "build" / "CHN.pkl").open("rb") as f:
        np.testing.assert_array_equal(table, pickle.load(f)[2015])
    assert store.countries() == ["AAA", "CHN"]
    assert (store.cache_dir / "CHN.npz").is_file()

    # Once cached, the mirror is not needed any more.
    offline = TableStore(dataclasses.replace(release, mirror_url="file:///nonexistent"), cache_root=tmp / "cache")
    np.testing.assert_array_equal(offline.get("CHN", 2015, data_dir=None), table)


def test_tampered_mirror_table_is_rejected(mirror):
    release, tmp = mirror
    (tmp / "mirror" / "AAA.npz").write_bytes(b"not the pinned bytes")
    with pytest.raises(TableNotFound, match="refusing"):
        TableStore(release, cache_root=tmp / "cache").get("AAA", 2014, data_dir=None)


def test_unpinned_manifest_is_rejected(mirror):
    release, tmp = mirror
    manifest = tmp / "mirror" / "MANIFEST.json"
    manifest.write_text(manifest.read_text().replace("AAA", "ZZZ"))
    with pytest.raises(TableNotFound, match="refusing"):
        TableStore(release, cache_root=tmp / "cache").countries()


def test_local_data_dir_wins_and_broken_pickles_fall_through(mirror):
    release, tmp = mirror
    local = tmp / "local"
    local.mkdir()
    with (local / "AAA.pkl").open("wb") as f:
        pickle.dump({2015: np.full((2, 2), 7.0)}, f)
    with (local / "CHN.pkl").open("wb") as f:
        pickle.dump([2014, 2015], f)  # like the broken TUN/VNM files
    store = TableStore(release, cache_root=tmp / "cache")

    assert store.get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert store.get("CHN", 2015, data_dir=local).shape == (2, 2)  # from the mirror


def test_missing_mirror_pin_explains_how_to_rebuild(tmp_path):
    release = dataclasses.replace(RELEASE, mirror_manifest_sha256=None)
    with pytest.raises(TableNotFound, match="python -m network_generator.oecd_icio build"):
        TableStore(release, cache_root=tmp_path).get("HUN", 2015, data_dir=None)


def test_unknown_year_and_country(mirror):
    release, tmp = mirror
    store = TableStore(release, cache_root=tmp / "cache")
    with pytest.raises(TableNotFound, match="no 1990 table"):
        store.get("AAA", 1990, data_dir=None)
    with pytest.raises(TableNotFound, match="not in"):
        store.get("XYZ", 2015, data_dir=None)


def test_generate_synthetic_network_end_to_end(tmp_path):
    from network_generator.pipeline import DegreeAcceptance, generate_synthetic_network

    result = generate_synthetic_network(
        "HUN", 2015, num_firms=3_000, acceptance=DegreeAcceptance.none(), seed=0, data_path=REPO / "data"
    )

    report = result.iot_report()
    assert result.weights.shape == (3_000, 3_000)
    assert report["num_edges"] > 0
    # Every industry pair that has edges is rescaled exactly to the target; what is missing is the uncovered flow.
    assert abs(report["relative_total_error"] + report["uncovered_share_of_flow"]) < 1e-4

    out = result.save(tmp_path / "run")
    run = json.loads((out / "run.json").read_text())
    assert run["country_code"] == "HUN" and run["seed"] == 0
    assert (out / "firms.csv").is_file()

    again = generate_synthetic_network(
        "HUN", 2015, num_firms=3_000, acceptance=DegreeAcceptance.none(), seed=0, data_path=REPO / "data"
    )
    assert again.weights._nnz() == result.weights._nnz()


def test_local_folder_is_only_used_for_its_own_release(mirror):
    release, tmp = mirror
    other = dataclasses.replace(release, release_id="oecd_icio_2025")
    local = tmp / "local"
    local.mkdir()
    with (local / "AAA.pkl").open("wb") as f:
        pickle.dump({2015: np.full((2, 2), 7.0)}, f)

    # No MANIFEST.json: a legacy folder, treated as the 2021 edition.
    assert TableStore(release, cache_root=tmp / "c1").get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert TableStore(other, cache_root=tmp / "c2").get("AAA", 2015, data_dir=local)[0, 0] != 7.0

    # A built folder declares its release.
    (local / "MANIFEST.json").write_text(json.dumps({"release_id": "oecd_icio_2025"}))
    assert TableStore(other, cache_root=tmp / "c3").get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert TableStore(release, cache_root=tmp / "c4").get("AAA", 2015, data_dir=local)[0, 0] != 7.0
