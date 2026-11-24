"""Turn OECD ICIO CSVs into the per-country tables the generator reads.

For each country and year the output is the industry x industry matrix of total intermediate use: entry (i, j) is what industry j of the country buys from industry i of *every* supplier country, domestic and imported, in millions of USD. Concretely, for country ``c``:

    Z_c[i, j] = sum over supplier entities s of ICIO[s_i, c_j]

Rows that are not industries (``TAXSUB``, ``VALU``, ``OUTPUT``) are dropped. Column entities listed in ``column_aggregates`` (China's and Mexico's processing-trade splits CN1/CN2 and MX1/MX2) are summed into their parent country. Supplier rows need no aggregation because the sum runs over all of them.

A release can set ``round_decimals`` to round every cell (in float64, before the float32 cast); the 2021 release rounds to 0.1 million USD to match the tables the paper was run on.

Each country is written as ``{ISO3}.pkl`` holding ``{year: float32 array (n_industries, n_industries)}``, the format ``data_iot.load_input_output_table`` expects. A ``MANIFEST.json`` next to the pickles records the release, the sha256 of every source bundle and CSV, the industry order and the sha256 of every output file.
"""

import datetime as _dt
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from network_generator.oecd_icio.archive import file_sha256

logger = logging.getLogger(__name__)

RELEASES_PATH = Path(__file__).with_name("releases.yaml")


@dataclass(frozen=True)
class Bundle:
    bundle_id: str
    url: str | None
    sha256: str | None


@dataclass(frozen=True)
class Release:
    release_id: str
    name: str
    citation: str
    csv_pattern: str
    column_aggregates: dict[str, str] = field(default_factory=dict)
    exclude_countries: tuple[str, ...] = ()
    bundles: tuple[Bundle, ...] = ()
    mirror_url: str | None = None
    mirror_manifest_sha256: str | None = None
    csv_sha256: dict[int, str] = field(default_factory=dict)
    """Per-year CSV pins, for editions whose bundles get unzipped on download (browsers do this) so bundle checksums can't be checked."""
    round_decimals: int | None = None
    """Round each table cell (millions USD) to this many decimals before storing. The 2021 release uses 1 to match the tables the paper was run on."""


def load_releases(path: Path = RELEASES_PATH) -> dict[str, Release]:
    raw = yaml.safe_load(path.read_text())
    releases = {}
    for release_id, entry in raw.items():
        mirror = entry.get("mirror") or {}
        releases[release_id] = Release(
            release_id=release_id,
            name=entry["name"],
            citation=entry["citation"],
            csv_pattern=entry["csv_pattern"],
            column_aggregates=dict(entry.get("column_aggregates") or {}),
            exclude_countries=tuple(entry.get("exclude_countries") or ()),
            bundles=tuple(Bundle(**b) for b in entry.get("bundles") or ()),
            mirror_url=mirror.get("url"),
            mirror_manifest_sha256=mirror.get("manifest_sha256"),
            csv_sha256={int(y): sha for y, sha in (entry.get("csv_sha256") or {}).items()},
            round_decimals=entry.get("round_decimals"),
        )
    return releases


def find_year_csvs(paths: list[Path], csv_pattern: str) -> dict[int, Path]:
    """Map year -> CSV for every file under ``paths`` whose name matches ``csv_pattern``.

    ``csv_pattern`` must define a ``year`` group. It is release specific on purpose: a generic "find the 4-digit token" rule breaks on names like ``ICIO2021_2014.csv``.
    """
    regex = re.compile(csv_pattern)
    found: dict[int, Path] = {}
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.csv"))
        for csv in candidates:
            match = regex.match(csv.name)
            if match is None:
                continue
            year = int(match.group("year"))
            if year in found and found[year] != csv:
                raise ValueError(f"two CSVs for {year}: {found[year]} and {csv}")
            found[year] = csv
    return found


def split_label(label: str) -> tuple[str, str] | None:
    """Split ``"<ENTITY>_<CODE>"`` on the first underscore; bare labels give None."""
    if "_" not in label:
        return None
    entity, code = label.split("_", 1)
    return entity, code


def industry_axis(labels: list[str], what: str) -> tuple[list[str], list[str]]:
    """Return ``(entities, industries)`` in first-seen order, checking every entity carries the same industries.

    ``what`` is used in error messages only.
    """
    per_entity: dict[str, list[str]] = {}
    for label in labels:
        parts = split_label(label)
        if parts is not None:
            per_entity.setdefault(parts[0], []).append(parts[1])
    if not per_entity:
        raise ValueError(f"no ENTITY_CODE labels on the {what} axis")
    entities = list(per_entity)
    industries = per_entity[entities[0]]
    for entity, codes in per_entity.items():
        if codes != industries:
            raise ValueError(
                f"{what} axis: {entity} has industries {codes}, expected {industries}"
            )
    return entities, industries


def total_use_by_country(
    csv: Path, release: Release
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read one year's ICIO CSV and return ``(industries, {country: Z})`` in float64."""
    df = pd.read_csv(csv, index_col=0)
    columns = [str(c) for c in df.columns]
    rows = [str(r) for r in df.index]

    # Columns hold industries and final-demand categories (HFCE, GFCF, ...).
    # Industries are the codes that also appear on the row axis.
    row_codes = {p[1] for r in rows if (p := split_label(r))}
    industry_columns = [c for c in columns if (p := split_label(c)) and p[1] in row_codes]
    col_entities, industries = industry_axis(industry_columns, "column")
    row_industry_labels = [
        r for r in rows if (p := split_label(r)) and p[1] in set(industries)
    ]
    _, row_industries = industry_axis(row_industry_labels, "row")
    if row_industries != industries:
        raise ValueError(f"{csv.name}: row and column industry lists differ")

    # Sum every supplier entity's rows by industry: (industries x all columns).
    body = df.loc[row_industry_labels, industry_columns].to_numpy(dtype=np.float64)
    row_industry_index = np.array(
        [industries.index(split_label(r)[1]) for r in row_industry_labels]
    )
    supplied = np.zeros((len(industries), body.shape[1]))
    np.add.at(supplied, row_industry_index, body)
    n = len(industries)

    countries: dict[str, np.ndarray] = {}
    for k, entity in enumerate(col_entities):
        block = supplied[:, k * n : (k + 1) * n]
        country = release.column_aggregates.get(entity, entity)
        if country in release.exclude_countries:
            continue
        countries[country] = countries.get(country, 0) + block
    return industries, countries


def build(
    release: Release,
    sources: list[Path],
    out_dir: Path,
    bundle_shas: dict[str, str] | None = None,
    years: list[int] | None = None,
    allow_unpinned: bool = False,
) -> dict:
    """Build ``{ISO3}.pkl`` files and ``MANIFEST.json`` in ``out_dir``; return the manifest.

    If the release pins CSV checksums, every CSV must match its pin (or ``allow_unpinned`` must be set, which only logs a warning).
    """
    csvs = find_year_csvs(sources, release.csv_pattern)
    if years is not None:
        missing = sorted(set(years) - set(csvs))
        if missing:
            raise ValueError(f"no CSV found for years {missing}")
        csvs = {y: csvs[y] for y in years}
    if not csvs:
        raise ValueError(f"no CSV matching {release.csv_pattern!r} under {sources}")

    csv_shas = {year: file_sha256(path) for year, path in csvs.items()}
    if release.csv_sha256:
        bad = [
            f"{csvs[y].name}: sha256 {sha} (pinned: {release.csv_sha256.get(y)})"
            for y, sha in sorted(csv_shas.items())
            if release.csv_sha256.get(y) != sha
        ]
        if bad:
            message = f"CSVs do not match the pins for {release.release_id}:\n  " + "\n  ".join(bad)
            if not allow_unpinned:
                raise ValueError(message)
            logger.warning(message)

    tables: dict[str, dict[int, np.ndarray]] = {}
    industries: list[str] | None = None
    csv_records = []
    for year in sorted(csvs):
        logger.info("processing %s", csvs[year].name)
        year_industries, by_country = total_use_by_country(csvs[year], release)
        if industries is None:
            industries = year_industries
        elif year_industries != industries:
            raise ValueError(f"{year}: industry list differs from earlier years")
        if tables and set(by_country) != set(tables):
            raise ValueError(f"{year}: country set differs from earlier years")
        for country, matrix in by_country.items():
            if release.round_decimals is not None:
                matrix = np.round(matrix, release.round_decimals)  # in float64, before the float32 cast
            tables.setdefault(country, {})[year] = matrix.astype(np.float32)
        csv_records.append(
            {"year": year, "file": csvs[year].name, "sha256": csv_shas[year]}
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for country in sorted(tables):
        path = out_dir / f"{country}.pkl"
        with path.open("wb") as f:
            pickle.dump(tables[country], f, protocol=4)
        outputs[path.name] = file_sha256(path)

    manifest = {
        "release_id": release.release_id,
        "release_name": release.name,
        "citation": release.citation,
        "built_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "recipe": "total intermediate use: Z_c[i, j] = sum_s ICIO[s_i, c_j], millions USD"
        + (f", rounded to {release.round_decimals} decimals" if release.round_decimals is not None else ""),
        "column_aggregates": release.column_aggregates,
        "excluded_countries": list(release.exclude_countries),
        "years": sorted(csvs),
        "industries": industries,
        "bundles": bundle_shas or {},
        "csvs": csv_records,
        "outputs": outputs,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def compare(built_dir: Path, reference_dir: Path, atol: float = 0.1) -> list[str]:
    """Compare every pickle in ``built_dir`` with its namesake in ``reference_dir``.

    Returns a list of problems (empty when everything matches within ``atol`` million USD). Only years present in the built file are checked, so a partial build can be compared with the full reference.
    """
    problems = []
    for built in sorted(built_dir.glob("*.pkl")):
        ref_path = reference_dir / built.name
        if not ref_path.exists():
            problems.append(f"{built.name}: missing from {reference_dir}")
            continue
        with built.open("rb") as f:
            new = pickle.load(f)
        with ref_path.open("rb") as f:
            old = pickle.load(f)
        if not isinstance(old, dict):
            problems.append(
                f"{built.name}: reference is a {type(old).__name__}, not a year -> matrix dict"
            )
            continue
        for year in sorted(new):
            if year not in old:
                problems.append(f"{built.name} {year}: missing from reference")
                continue
            a, b = np.asarray(new[year]), np.asarray(old[year])
            if a.shape != b.shape:
                problems.append(f"{built.name} {year}: shape {a.shape} vs {b.shape}")
                continue
            diff = float(np.abs(a - b).max())
            if diff > atol:
                problems.append(f"{built.name} {year}: max |diff| {diff:.4g}")
    return problems
