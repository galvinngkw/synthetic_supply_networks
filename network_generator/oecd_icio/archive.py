"""Download, hash, cache and extract OECD ICIO source bundles.

Bundles are cached content-addressed by sha256, so a URL that starts serving different bytes never silently replaces what a build was made from.
"""

import hashlib
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; synthetic-supply-networks-data/0.1)"


def default_cache_root() -> Path:
    """Cache directory for raw bundles; override with ``SSN_DATA_CACHE``."""
    env = os.environ.get("SSN_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "synthetic_supply_networks" / "oecd_icio"


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> str:
    """Stream ``url`` to ``dest`` and return its sha256.

    OECD's file servers often refuse non-browser clients (HTTP 403). When that happens, download the bundle in a browser and pass it to the build with ``--bundle``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    logger.info("downloading %s", url)
    with urllib.request.urlopen(request, timeout=300) as resp, dest.open("wb") as f:
        while chunk := resp.read(chunk_size):
            hasher.update(chunk)
            f.write(chunk)
    return hasher.hexdigest()


def cache_bundle(source: Path, sha256: str, cache_root: Path) -> Path:
    """Copy ``source`` into ``cache_root/<sha256>/`` unless it is already there."""
    target = cache_root / sha256 / source.name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        logger.info("cached %s", target)
    return target


def extract(bundle: Path) -> Path:
    """Return a directory holding the bundle's CSVs.

    A ZIP is extracted next to itself (once); a directory is returned as is; a single CSV returns its parent.
    """
    if bundle.is_dir():
        return bundle
    if zipfile.is_zipfile(bundle):
        out = bundle.with_suffix(".extracted")
        if not out.exists():
            tmp = bundle.with_suffix(".extracting")
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(bundle) as zf:
                zf.extractall(tmp)
            tmp.rename(out)
        return out
    return bundle.parent
