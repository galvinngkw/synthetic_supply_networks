import dataclasses
import json
import pickle
import shutil

import numpy as np
import pytest
import torch

from network_generator.config import CONF_PATH, _find_conf_path
from network_generator.oecd_icio import TableNotFound, TableStore
from tests.conftest import REPO, make_mirror


def _pickle(path, obj):
    with path.open("wb") as f:
        pickle.dump(obj, f)


# -- mirror and cache ------------------------------------------------------------


def test_mirror_download_matches_build_and_is_cached(mirror):
    release, tmp = mirror
    store = TableStore(release, cache_root=tmp / "cache")

    table = store.get("CHN", 2015)

    with (tmp / "build0" / "CHN.pkl").open("rb") as f:
        np.testing.assert_array_equal(table, pickle.load(f)[2015])
    assert store.countries() == ["AAA", "CHN"]
    assert (store.cache_dir / "CHN.npz").is_file()

    # Once cached, the mirror is not needed any more.
    offline = TableStore(dataclasses.replace(release, mirror_url="file:///nonexistent"), cache_root=tmp / "cache")
    np.testing.assert_array_equal(offline.get("CHN", 2015), table)


def test_repinned_mirror_replaces_stale_cached_tables(release, tmp_path):
    first = make_mirror(release, tmp_path, seed=0)
    old = TableStore(first, cache_root=tmp_path / "cache").get("AAA", 2015)

    # The mirror is republished with different tables and releases.yaml gets the new pin.
    shutil.rmtree(tmp_path / "mirror")
    second = make_mirror(release, tmp_path, seed=1)
    new = TableStore(second, cache_root=tmp_path / "cache").get("AAA", 2015)

    with (tmp_path / "build1" / "AAA.pkl").open("rb") as f:
        np.testing.assert_array_equal(new, pickle.load(f)[2015])
    assert not np.array_equal(new, old)


def test_tampered_mirror_table_is_rejected(mirror):
    release, tmp = mirror
    (tmp / "mirror" / "AAA.npz").write_bytes(b"not the pinned bytes")
    with pytest.raises(TableNotFound, match="refusing"):
        TableStore(release, cache_root=tmp / "cache").get("AAA", 2014)


def test_unpinned_manifest_is_rejected(mirror):
    release, tmp = mirror
    manifest = tmp / "mirror" / "MANIFEST.json"
    manifest.write_text(manifest.read_text().replace("AAA", "ZZZ"))
    with pytest.raises(TableNotFound, match="refusing"):
        TableStore(release, cache_root=tmp / "cache").countries()


def test_missing_mirror_pin_explains_how_to_rebuild(release, tmp_path):
    release = dataclasses.replace(release, mirror_manifest_sha256=None)
    with pytest.raises(TableNotFound, match="python -m network_generator.oecd_icio build"):
        TableStore(release, cache_root=tmp_path).get("HUN", 2015)


def test_unknown_year_and_country(mirror):
    release, tmp = mirror
    store = TableStore(release, cache_root=tmp / "cache")
    with pytest.raises(TableNotFound, match="no 1990 table"):
        store.get("AAA", 1990)
    with pytest.raises(TableNotFound, match="not in"):
        store.get("XYZ", 2015)


# -- local folders -----------------------------------------------------------------


def test_local_folder_wins_and_absent_countries_come_from_the_mirror(mirror):
    release, tmp = mirror
    local = tmp / "local"
    local.mkdir()
    _pickle(local / "AAA.pkl", {2015: np.full((2, 2), 7.0)})
    store = TableStore(release, cache_root=tmp / "cache")

    assert store.get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert store.get("CHN", 2015, data_dir=local).shape == (2, 2)  # no CHN.pkl there: from the mirror


def test_broken_or_incomplete_local_table_is_an_error_not_replaced(mirror):
    release, tmp = mirror
    local = tmp / "local"
    local.mkdir()
    _pickle(local / "AAA.pkl", {2014: np.zeros((2, 2))})
    _pickle(local / "CHN.pkl", [2014, 2015])  # like the old broken TUN/VNM files
    store = TableStore(release, cache_root=tmp / "cache")

    with pytest.raises(TableNotFound, match="no 2015 table"):
        store.get("AAA", 2015, data_dir=local)
    with pytest.raises(TableNotFound, match="not a year -> matrix dict"):
        store.get("CHN", 2015, data_dir=local)
    with pytest.raises(TableNotFound, match="does not exist"):
        store.get("AAA", 2015, data_dir=tmp / "typo")


def test_local_folder_is_only_used_for_its_own_release(mirror):
    release, tmp = mirror
    other = dataclasses.replace(release, release_id="oecd_icio_2025")
    local = tmp / "local"
    local.mkdir()
    _pickle(local / "AAA.pkl", {2015: np.full((2, 2), 7.0)})

    # No MANIFEST.json: a legacy folder, treated as the 2021 edition.
    assert TableStore(release, cache_root=tmp / "c1").get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert TableStore(other, cache_root=tmp / "c2").get("AAA", 2015, data_dir=local)[0, 0] != 7.0

    # A built folder declares its release.
    (local / "MANIFEST.json").write_text(json.dumps({"release_id": "oecd_icio_2025"}))
    assert TableStore(other, cache_root=tmp / "c3").get("AAA", 2015, data_dir=local)[0, 0] == 7.0
    assert TableStore(release, cache_root=tmp / "c4").get("AAA", 2015, data_dir=local)[0, 0] != 7.0


# -- configuration -----------------------------------------------------------------


def test_conf_path_uses_checkout_conf_or_packaged_copy(tmp_path):
    package_dir = tmp_path / "site-packages" / "network_generator"
    assert _find_conf_path(package_dir, in_checkout=True) == tmp_path / "site-packages" / "conf"
    assert _find_conf_path(package_dir, in_checkout=False) == package_dir / "conf"
    # In this checkout the defaults come from the repo's top-level conf/.
    assert CONF_PATH == REPO / "conf" and (CONF_PATH / "network" / "default.yaml").is_file()


# -- end to end ----------------------------------------------------------------------


def test_degree_draws_no_graph_can_realise_are_rejected():
    from network_generator.pipeline import _valid_degrees

    assert _valid_degrees(torch.tensor([1, 2, 3]), n=4)
    assert not _valid_degrees(torch.tensor([1, 2, 4]), n=4)  # degree n needs a self-loop
    assert not _valid_degrees(torch.tensor([1, -1, 2]), n=4)
    assert not _valid_degrees(torch.tensor([0, 0, 0]), n=4)
    assert not _valid_degrees(torch.tensor([], dtype=torch.int64), n=4)


def test_generate_synthetic_network_end_to_end(tmp_path):
    from network_generator import DegreeAcceptance, generate_synthetic_network

    def run():
        return generate_synthetic_network(
            "HUN", 2015, num_firms=3_000, acceptance=DegreeAcceptance.none(), seed=0, data_path=REPO / "data"
        )

    result = run()
    report = result.iot_report()
    assert result.weights.shape == (3_000, 3_000)
    assert report["num_edges"] > 0
    # Every industry pair that has edges is rescaled exactly to the target; what is missing is the uncovered flow.
    assert abs(report["relative_total_error"] + report["uncovered_share_of_flow"]) < 1e-4

    out = result.save(tmp_path / "run")
    run_info = json.loads((out / "run.json").read_text())
    assert run_info["country_code"] == "HUN" and run_info["seed"] == 0
    assert (out / "edges.csv").is_file() and (out / "firms.csv").is_file()
    with pytest.raises(ValueError, match="edge_format"):
        result.save(tmp_path / "bad", edge_format="xlsx")

    # Same seed, same network: identical edges, weights and firm assignment.
    again = run()
    a, b = result.weights.coalesce(), again.weights.coalesce()
    assert torch.equal(a.indices(), b.indices())
    assert torch.equal(a.values(), b.values())
    assert torch.equal(result.firm_to_industry, again.firm_to_industry)
