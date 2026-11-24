import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_generator.oecd_icio import build, load_releases
from network_generator.oecd_icio.archive import file_sha256
from network_generator.oecd_icio.store import package_for_mirror

REPO = Path(__file__).resolve().parents[1]


def write_toy_icio(path: Path, seed: int = 0) -> None:
    """Tiny ICIO with 2 industries, countries AAA, CHN (split into CN1/CN2) and ROW.

    Rows: suppliers incl. the CN1/CN2 split, plus TAXSUB, VALU and OUTPUT. Columns: industries plus one final-demand column per country.
    """
    inds = ["A", "B"]
    entities = ["AAA", "CHN", "CN1", "CN2", "ROW"]
    rows = [f"{e}_{i}" for e in entities for i in inds]
    rows += [f"{e}_TAXSUB" for e in ["AAA", "ROW"]] + ["VALU", "OUTPUT"]
    cols = [f"{e}_{i}" for e in entities for i in inds]
    cols += [f"{e}_HFCE" for e in ["AAA", "ROW"]]
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(1, 10, (len(rows), len(cols))).astype(float), rows, cols)
    df[[c for c in cols if c.startswith("CHN_")]] = 0.0  # as in the 2021 edition
    df.to_csv(path)


@pytest.fixture
def pinned_release():
    """The real 2021 release, with its CSV pins."""
    return load_releases()["oecd_icio_2021"]


@pytest.fixture
def release(pinned_release):
    """The 2021 release without CSV pins: toy CSVs can't match the real ones."""
    return dataclasses.replace(pinned_release, csv_sha256={})


@pytest.fixture
def toy_icio():
    return write_toy_icio


def make_mirror(release, root: Path, seed: int = 0):
    """Build toy tables, package them as mirror assets under ``root/mirror`` and return the release pinned to them."""
    src = root / f"src{seed}"
    src.mkdir(parents=True)
    for year in (2014, 2015):
        write_toy_icio(src / f"ICIO2021_{year}.csv", seed=seed)
    build(release, [src], root / f"build{seed}")
    package_for_mirror(root / f"build{seed}", root / "mirror")
    pin = file_sha256(root / "mirror" / "MANIFEST.json")
    return dataclasses.replace(release, mirror_url=(root / "mirror").as_uri(), mirror_manifest_sha256=pin)


@pytest.fixture
def mirror(release, tmp_path):
    """A toy build packaged as mirror assets, served from a file:// URL. Returns (pinned release, tmp_path)."""
    return make_mirror(release, tmp_path), tmp_path
