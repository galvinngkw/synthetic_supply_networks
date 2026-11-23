"""Find a country's input-output table wherever it is: local folder, cache, or mirror.

Lookup order for ``get_table(country, year)``:

1. ``data_dir`` (default ``data/``): ``{ISO3}.pkl`` as written by the build, if it holds the year and the folder belongs to the requested release (its ``MANIFEST.json``; a folder without one counts as ``LEGACY_RELEASE``). Kept so existing checkouts and hand-made tables keep working.
2. The local cache, ``<cache>/tables/<release>/{ISO3}.npz``.
3. The release's mirror. The mirror's ``MANIFEST.json`` is checked against the sha256 pinned in ``releases.yaml``, and each table against the sha256 listed in that manifest, so a pinned commit of this repo fixes the exact numbers it runs on. Mirror tables are ``.npz`` read with ``allow_pickle=False``: nothing downloaded is ever unpickled.

If none of these has the table, the error says how to rebuild it from the raw OECD files.
"""

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np

from network_generator.oecd_icio.archive import default_cache_root, download, file_sha256
from network_generator.oecd_icio.build import Release, load_releases

DEFAULT_RELEASE = "oecd_icio_2021"
# Release assumed for a data folder without MANIFEST.json (the tables committed before the pipeline existed).
LEGACY_RELEASE = "oecd_icio_2021"
MIRROR_ENV = "SSN_ICIO_MIRROR"


class TableNotFound(LookupError):
    pass


def _year_key(year: int) -> str:
    return f"y{year}"


def write_npz(path: Path, tables: dict[int, np.ndarray]) -> None:
    """Write ``{year: matrix}`` as an ``.npz`` with keys ``y<year>``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **{_year_key(y): np.asarray(m) for y, m in tables.items()})


def read_npz(path: Path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {int(k[1:]): z[k] for k in z.files}


def local_release(data_dir: Path) -> str:
    """Release a local data folder was built from, per its MANIFEST.json (legacy folders have none)."""
    manifest = data_dir / "MANIFEST.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())["release_id"]
    return LEGACY_RELEASE


def _read_local_pickle(path: Path, year: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        tables = pickle.load(f)
    if not isinstance(tables, dict) or year not in tables:
        return None
    return np.asarray(tables[year])


class TableStore:
    """Resolve, download and cache the tables of one release."""

    def __init__(
        self,
        release: Release | str = DEFAULT_RELEASE,
        cache_root: Path | None = None,
        mirror_url: str | None = None,
    ):
        self.release = load_releases()[release] if isinstance(release, str) else release
        self.cache_dir = (cache_root or default_cache_root()) / "tables" / self.release.release_id
        self.mirror_url = (mirror_url or os.environ.get(MIRROR_ENV) or self.release.mirror_url or "").rstrip("/") or None
        self._manifest: dict | None = None

    # -- mirror -----------------------------------------------------------------

    def manifest(self) -> dict:
        """The mirror manifest, downloaded once, checked against the pin in releases.yaml."""
        if self._manifest is not None:
            return self._manifest
        cached = self.cache_dir / "MANIFEST.json"
        pin = self.release.mirror_manifest_sha256
        if not (cached.is_file() and pin and file_sha256(cached) == pin):
            if self.mirror_url is None:
                raise TableNotFound(self._rebuild_hint("this release has no mirror configured"))
            if pin is None:
                raise TableNotFound(
                    self._rebuild_hint(
                        f"the mirror manifest for {self.release.release_id} is not pinned in releases.yaml yet"
                    )
                )
            self._fetch(f"{self.mirror_url}/MANIFEST.json", cached, pin)
        self._manifest = json.loads(cached.read_text())
        return self._manifest

    def _fetch(self, url: str, dest: Path, sha256: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            tmp_path = Path(tmp) / dest.name
            try:
                got = download(url, tmp_path)
            except OSError as err:
                raise TableNotFound(self._rebuild_hint(f"downloading {url} failed ({err})")) from err
            if got != sha256:
                raise TableNotFound(
                    f"{url} has sha256 {got}, expected {sha256}. The mirror no longer serves the pinned file; refusing to use it."
                )
            shutil.move(tmp_path, dest)

    def _cached_tables(self, country: str) -> dict[int, np.ndarray] | None:
        path = self.cache_dir / f"{country}.npz"
        if not path.is_file():
            entry = self.manifest()["tables"].get(f"{country}.npz")
            if entry is None:
                return None
            self._fetch(f"{self.mirror_url}/{country}.npz", path, entry)
        return read_npz(path)

    # -- public -----------------------------------------------------------------

    def countries(self) -> list[str]:
        return sorted(name.removesuffix(".npz") for name in self.manifest()["tables"])

    def years(self) -> list[int]:
        return list(self.manifest()["years"])

    def industries(self) -> list[str]:
        return list(self.manifest()["industries"])

    def get(self, country: str, year: int, data_dir: Path | str | None = "data") -> np.ndarray:
        """Industry x industry total intermediate use for ``country`` in ``year``, millions USD."""
        country = country.upper()
        # A local folder is only used for the release it was built from, so asking for another edition never silently returns this one's tables.
        if data_dir is not None and Path(data_dir).is_dir() and local_release(Path(data_dir)) == self.release.release_id:
            local = _read_local_pickle(Path(data_dir) / f"{country}.pkl", year)
            if local is not None:
                return local
        tables = self._cached_tables(country)
        if tables is None:
            raise TableNotFound(
                f"{country} is not in {self.release.release_id}. Available: {', '.join(self.countries())}"
            )
        if year not in tables:
            raise TableNotFound(
                f"{self.release.release_id} has no {year} table for {country}; years available: {min(tables)}-{max(tables)}"
            )
        return tables[year]

    def _rebuild_hint(self, reason: str) -> str:
        return (
            f"Cannot get {self.release.release_id} tables: {reason}.\n"
            "Rebuild them from the raw OECD files instead: download the CSV bundles from "
            "http://oe.cd/icio in a browser, then run\n"
            f"    python -m network_generator.oecd_icio build --release {self.release.release_id} "
            "--bundle <zip> [--bundle <zip> ...] --out data"
        )


def get_table(
    country: str,
    year: int,
    release: str = DEFAULT_RELEASE,
    data_dir: Path | str | None = "data",
) -> np.ndarray:
    """Shortcut for ``TableStore(release).get(country, year, data_dir)``."""
    return TableStore(release).get(country, year, data_dir=data_dir)


def package_for_mirror(build_dir: Path, out_dir: Path) -> dict:
    """Turn a build folder (``{ISO3}.pkl`` + ``MANIFEST.json``) into mirror assets.

    Writes ``{ISO3}.npz`` for every table and a mirror ``MANIFEST.json`` that keeps the build's provenance (source bundle and CSV checksums) and lists the sha256 of every ``.npz``. Upload everything in ``out_dir`` to the mirror, then pin ``sha256(MANIFEST.json)`` as ``mirror.manifest_sha256`` in ``releases.yaml``.
    """
    build_manifest = json.loads((build_dir / "MANIFEST.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for pkl in sorted(build_dir.glob("*.pkl")):
        with pkl.open("rb") as f:
            data = pickle.load(f)
        npz = out_dir / f"{pkl.stem}.npz"
        write_npz(npz, data)
        tables[npz.name] = file_sha256(npz)
    manifest = {
        key: build_manifest[key]
        for key in ("release_id", "release_name", "citation", "built_at", "recipe", "column_aggregates", "excluded_countries", "years", "industries", "bundles", "csvs")
    }
    manifest["tables"] = tables
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
