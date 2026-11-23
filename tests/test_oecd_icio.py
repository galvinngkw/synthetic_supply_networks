import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_generator.oecd_icio import build, compare, load_releases, total_use_by_country
from network_generator.oecd_icio.build import find_year_csvs

REPO = Path(__file__).resolve().parents[1]
import dataclasses

PINNED_RELEASE = load_releases()["oecd_icio_2021"]
# Toy CSVs can't match the real pins, so unit tests use the release without them.
RELEASE = dataclasses.replace(PINNED_RELEASE, csv_sha256={})


def write_toy_icio(path: Path) -> None:
    """Tiny ICIO with 2 industries, countries AAA, CHN (split into CN1/CN2) and ROW.

    Rows: suppliers incl. the CN1/CN2 split, plus TAXSUB, VALU and OUTPUT. Columns: industries plus one final-demand column per country.
    """
    inds = ["A", "B"]
    row_entities = ["AAA", "CHN", "CN1", "CN2", "ROW"]
    col_entities = ["AAA", "CHN", "CN1", "CN2", "ROW"]
    rows = [f"{e}_{i}" for e in row_entities for i in inds]
    rows += [f"{e}_TAXSUB" for e in ["AAA", "ROW"]] + ["VALU", "OUTPUT"]
    cols = [f"{e}_{i}" for e in col_entities for i in inds]
    cols += [f"{e}_HFCE" for e in ["AAA", "ROW"]]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(1, 10, (len(rows), len(cols))).astype(float), rows, cols)
    df[[c for c in cols if c.startswith("CHN_")]] = 0.0  # as in the 2021 edition
    df.to_csv(path)


def expected_total_use(df: pd.DataFrame, country_cols: list[str]) -> np.ndarray:
    out = np.zeros((2, 2))
    for e in ["AAA", "CHN", "CN1", "CN2", "ROW"]:
        for prefix in country_cols:
            out += df.loc[[f"{e}_A", f"{e}_B"], [f"{prefix}_A", f"{prefix}_B"]].to_numpy()
    return out


def test_total_use_sums_all_suppliers_and_aggregates_split_columns(tmp_path):
    csv = tmp_path / "ICIO2021_2014.csv"
    write_toy_icio(csv)
    df = pd.read_csv(csv, index_col=0)

    industries, countries = total_use_by_country(csv, RELEASE)

    assert industries == ["A", "B"]
    assert set(countries) == {"AAA", "CHN"}  # ROW excluded, CN1/CN2 folded into CHN
    np.testing.assert_allclose(countries["AAA"], expected_total_use(df, ["AAA"]))
    np.testing.assert_allclose(
        countries["CHN"], expected_total_use(df, ["CHN", "CN1", "CN2"])
    )


def test_build_writes_loader_format_and_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for year in (2014, 2015):
        write_toy_icio(src / f"ICIO2021_{year}.csv")
    (src / "ICIO2021_2014_pivot.csv").write_text("ignored")  # not matched by csv_pattern

    manifest = build(RELEASE, [src], tmp_path / "out")

    assert manifest["years"] == [2014, 2015]
    assert set(manifest["outputs"]) == {"AAA.pkl", "CHN.pkl"}
    with (tmp_path / "out" / "AAA.pkl").open("rb") as f:
        table = pickle.load(f)
    assert set(table) == {2014, 2015}
    assert table[2014].dtype == np.float32 and table[2014].shape == (2, 2)
    assert compare(tmp_path / "out", tmp_path / "out") == []


def test_year_pattern_ignores_edition_year(tmp_path):
    (tmp_path / "ICIO2021_1995.csv").write_text("")
    assert list(find_year_csvs([tmp_path], RELEASE.csv_pattern)) == [1995]


@pytest.mark.skipif(
    "ICIO2021_CSV_DIR" not in os.environ,
    reason="set ICIO2021_CSV_DIR to a folder of OECD ICIO 2021 CSVs",
)
def test_reproduces_committed_tables(tmp_path):
    """Rebuild from the real OECD CSVs and check the result equals data/*.pkl exactly."""
    build(PINNED_RELEASE, [Path(os.environ["ICIO2021_CSV_DIR"])], tmp_path)
    # data/ is built by this pipeline, so a rebuild must reproduce it exactly.
    assert compare(tmp_path, REPO / "data", atol=0.0) == []


def test_2025_release_is_registered_and_label_agnostic(tmp_path):
    """The 2025 edition uses different industry codes and bare TLS/VA/OUT rows; the build must not care."""
    release = load_releases()["oecd_icio_2025"]
    csv = tmp_path / "2016.csv"
    write_toy_icio(csv)
    text = csv.read_text()
    for old, new in [("_A", "_A01"), ("_B", "_C24A"), ("VALU", "VA"), ("OUTPUT", "OUT"), ("TAXSUB", "TLS")]:
        text = text.replace(old, new)
    csv.write_text(text)

    assert list(find_year_csvs([tmp_path], release.csv_pattern)) == [2016]
    industries, countries = total_use_by_country(csv, release)
    assert industries == ["A01", "C24A"]
    assert set(countries) == {"AAA", "CHN"}
    assert all(b.sha256 for b in release.bundles)


def test_csv_pins_are_enforced(tmp_path):
    csv = tmp_path / "ICIO2021_2014.csv"
    write_toy_icio(csv)
    pinned = dataclasses.replace(RELEASE, csv_sha256={2014: "0" * 64})
    with pytest.raises(ValueError, match="do not match the pins"):
        build(pinned, [tmp_path], tmp_path / "out")
    assert build(pinned, [tmp_path], tmp_path / "out", allow_unpinned=True)["years"] == [2014]
