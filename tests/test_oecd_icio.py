import dataclasses
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_generator.oecd_icio import __main__ as cli
from network_generator.oecd_icio import build, compare, load_releases, total_use_by_country
from network_generator.oecd_icio.archive import file_sha256
from network_generator.oecd_icio.build import Bundle, find_year_csvs
from tests.conftest import REPO


def expected_total_use(df: pd.DataFrame, country_cols: list[str]) -> np.ndarray:
    out = np.zeros((2, 2))
    for e in ["AAA", "CHN", "CN1", "CN2", "ROW"]:
        for prefix in country_cols:
            out += df.loc[[f"{e}_A", f"{e}_B"], [f"{prefix}_A", f"{prefix}_B"]].to_numpy()
    return out


def test_total_use_sums_all_suppliers_and_aggregates_split_columns(tmp_path, release, toy_icio):
    csv = tmp_path / "ICIO2021_2014.csv"
    toy_icio(csv)
    df = pd.read_csv(csv, index_col=0)

    industries, countries = total_use_by_country(csv, release)

    assert industries == ["A", "B"]
    assert set(countries) == {"AAA", "CHN"}  # ROW excluded, CN1/CN2 folded into CHN
    np.testing.assert_allclose(countries["AAA"], expected_total_use(df, ["AAA"]))
    np.testing.assert_allclose(countries["CHN"], expected_total_use(df, ["CHN", "CN1", "CN2"]))


def test_build_writes_loader_format_and_manifest(tmp_path, release, toy_icio):
    src = tmp_path / "src"
    src.mkdir()
    for year in (2014, 2015):
        toy_icio(src / f"ICIO2021_{year}.csv")
    (src / "ICIO2021_2014_pivot.csv").write_text("ignored")  # not matched by csv_pattern

    manifest = build(release, [src], tmp_path / "out")

    assert manifest["years"] == [2014, 2015]
    assert set(manifest["outputs"]) == {"AAA.pkl", "CHN.pkl"}
    with (tmp_path / "out" / "AAA.pkl").open("rb") as f:
        table = pickle.load(f)
    assert set(table) == {2014, 2015}
    assert table[2014].dtype == np.float32 and table[2014].shape == (2, 2)
    assert compare(tmp_path / "out", tmp_path / "out") == []


def test_rounding_is_applied_before_float32(tmp_path, release, toy_icio):
    csv = tmp_path / "ICIO2021_2014.csv"
    toy_icio(csv)
    df = pd.read_csv(csv, index_col=0)
    df.iloc[0, 0] += 0.26  # AAA_A -> AAA_A no longer an integer
    df.to_csv(csv)

    build(dataclasses.replace(release, round_decimals=1), [tmp_path], tmp_path / "out")

    with (tmp_path / "out" / "AAA.pkl").open("rb") as f:
        cell = float(pickle.load(f)[2014][0, 0])
    assert cell == pytest.approx(round(expected_total_use(df, ["AAA"])[0, 0], 1), abs=1e-5)


def test_year_pattern_ignores_edition_year(tmp_path, release):
    (tmp_path / "ICIO2021_1995.csv").write_text("")
    assert list(find_year_csvs([tmp_path], release.csv_pattern)) == [1995]


@pytest.mark.skipif(
    "ICIO2021_CSV_DIR" not in os.environ,
    reason="set ICIO2021_CSV_DIR to a folder of OECD ICIO 2021 CSVs",
)
def test_reproduces_committed_tables(tmp_path, pinned_release):
    """Rebuild from the real OECD CSVs and check the result equals data/*.pkl exactly."""
    build(pinned_release, [Path(os.environ["ICIO2021_CSV_DIR"])], tmp_path)
    # data/ is built by this pipeline, so a rebuild must reproduce it exactly.
    assert compare(tmp_path, REPO / "data", atol=0.0) == []


def test_2025_release_is_registered_and_label_agnostic(tmp_path, toy_icio):
    """The 2025 edition uses different industry codes and bare TLS/VA/OUT rows; the build must not care."""
    release = load_releases()["oecd_icio_2025"]
    csv = tmp_path / "2016.csv"
    toy_icio(csv)
    text = csv.read_text()
    for old, new in [("_A", "_A01"), ("_B", "_C24A"), ("VALU", "VA"), ("OUTPUT", "OUT"), ("TAXSUB", "TLS")]:
        text = text.replace(old, new)
    csv.write_text(text)

    assert list(find_year_csvs([tmp_path], release.csv_pattern)) == [2016]
    industries, countries = total_use_by_country(csv, release)
    assert industries == ["A01", "C24A"]
    assert set(countries) == {"AAA", "CHN"}
    assert all(b.sha256 for b in release.bundles)


def test_csv_pins_are_enforced(tmp_path, release, toy_icio):
    toy_icio(tmp_path / "ICIO2021_2014.csv")
    pinned = dataclasses.replace(release, csv_sha256={2014: "0" * 64})
    with pytest.raises(ValueError, match="do not match the pins"):
        build(pinned, [tmp_path], tmp_path / "out")
    assert build(pinned, [tmp_path], tmp_path / "out", allow_unpinned=True)["years"] == [2014]


# -- command line --------------------------------------------------------------


def test_cli_build_and_compare(tmp_path, toy_icio):
    src = tmp_path / "src"
    src.mkdir()
    toy_icio(src / "ICIO2021_2014.csv")

    # The real 2021 pins can't match a toy CSV: the build refuses it unless told otherwise.
    with pytest.raises(SystemExit, match="do not match the pins"):
        cli.main(["build", "--bundle", str(src), "--out", str(tmp_path / "a")])
    assert cli.main(["build", "--bundle", str(src), "--out", str(tmp_path / "a"), "--allow-unpinned"]) == 0
    assert (tmp_path / "a" / "MANIFEST.json").is_file()

    assert cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "a")]) == 0
    toy_icio(src / "ICIO2021_2014.csv", seed=1)
    cli.main(["build", "--bundle", str(src), "--out", str(tmp_path / "b"), "--allow-unpinned"])
    assert cli.main(["compare", str(tmp_path / "b"), str(tmp_path / "a")]) == 1


def test_cli_reuses_a_cached_bundle_instead_of_downloading(tmp_path, release, toy_icio, monkeypatch):
    csv = tmp_path / "ICIO2021_2014.csv"
    toy_icio(csv)
    bundle_zip = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.write(csv, csv.name)
    sha = file_sha256(bundle_zip)
    with_url = dataclasses.replace(
        release, bundles=(Bundle(bundle_id="2014", url="https://example.invalid/b.zip", sha256=sha),)
    )
    monkeypatch.setattr(cli, "load_releases", lambda: {"oecd_icio_2021": with_url})
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.write_bytes(bundle_zip.read_bytes())
        return sha

    monkeypatch.setattr(cli, "download", fake_download)
    args = ["build", "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")]

    assert cli.main(args) == 0
    assert cli.main(args) == 0
    assert calls == ["https://example.invalid/b.zip"]  # the second run used the cache
