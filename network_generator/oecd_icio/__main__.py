"""Command line entry point: ``python -m network_generator.oecd_icio``.

Build the country tables from pinned OECD bundles (downloaded, or passed with ``--bundle`` when OECD blocks scripted downloads):

    python -m network_generator.oecd_icio build --bundle ~/Downloads/ICIO2021_2015-2018.zip --out data

Check a build against the committed tables:

    python -m network_generator.oecd_icio compare build/icio data

Download tables from the pinned mirror into the local cache (all of them if no country is given):

    python -m network_generator.oecd_icio fetch HUN DEU

Turn a build into mirror assets (``{ISO3}.npz`` + ``MANIFEST.json``) ready to upload:

    python -m network_generator.oecd_icio package build/icio dist/mirror
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from network_generator.oecd_icio.archive import (
    cache_bundle,
    default_cache_root,
    download,
    extract,
    file_sha256,
)
from network_generator.oecd_icio.build import build, compare, load_releases
from network_generator.oecd_icio.store import (
    DEFAULT_RELEASE,
    TableNotFound,
    TableStore,
    package_for_mirror,
)


def _resolve_sources(release, local_bundles, cache_root, allow_unpinned):
    """Return ``(csv_roots, {bundle name: sha256})`` after pin checks."""
    pinned = {b.sha256: b.bundle_id for b in release.bundles if b.sha256}
    sources, shas = [], {}

    def check(name, sha):
        if sha in pinned:
            return
        message = (
            f"{name} has sha256 {sha}, which is not pinned in releases.yaml for "
            f"{release.release_id}. Add it under the matching bundle to accept it."
        )
        if not allow_unpinned:
            raise SystemExit(message + " (or pass --allow-unpinned)")
        logging.warning(message)

    if local_bundles:
        for path in local_bundles:
            path = path.expanduser().resolve()
            if path.is_dir() or path.suffix.lower() == ".csv":
                # Unpacked data can't be matched against bundle pins; the manifest still records every CSV's sha256.
                sources.append(path)
                continue
            sha = file_sha256(path)
            check(path.name, sha)
            shas[path.name] = sha
            sources.append(extract(cache_bundle(path, sha, cache_root)))
        return sources, shas

    downloadable = [b for b in release.bundles if b.url]
    if not downloadable:
        raise SystemExit(
            f"{release.release_id} has no download URLs. Download the bundles in a "
            "browser and pass them with --bundle."
        )
    for bundle in downloadable:
        cached = sorted((cache_root / bundle.sha256).glob("*.zip")) if bundle.sha256 else []
        if cached and file_sha256(cached[0]) == bundle.sha256:
            logging.info("using cached %s (%s)", bundle.bundle_id, cached[0])
            shas[bundle.bundle_id] = bundle.sha256
            sources.append(extract(cached[0]))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / f"{bundle.bundle_id}.zip"
            try:
                sha = download(bundle.url, tmp_path)
            except OSError as err:
                raise SystemExit(
                    f"downloading {bundle.bundle_id} failed ({err}). OECD often blocks "
                    "scripted downloads: fetch it in a browser and pass --bundle."
                ) from err
            if bundle.sha256 and sha != bundle.sha256:
                raise SystemExit(
                    f"{bundle.bundle_id}: downloaded sha256 {sha} but releases.yaml "
                    f"pins {bundle.sha256}. The source changed; review before re-pinning."
                )
            check(bundle.bundle_id, sha)
            shas[bundle.bundle_id] = sha
            sources.append(extract(cache_bundle(tmp_path, sha, cache_root)))
    return sources, shas


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m network_generator.oecd_icio")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build {ISO3}.pkl tables and MANIFEST.json")
    p_build.add_argument("--release", default=DEFAULT_RELEASE)
    p_build.add_argument(
        "--bundle",
        action="append",
        type=Path,
        default=[],
        help="local bundle ZIP, extracted directory or single CSV (repeatable)",
    )
    p_build.add_argument("--out", type=Path, default=Path("data"))
    p_build.add_argument("--years", type=int, nargs="+")
    p_build.add_argument("--cache", type=Path, default=None)
    p_build.add_argument("--allow-unpinned", action="store_true")

    p_cmp = sub.add_parser("compare", help="compare built tables with reference ones")
    p_cmp.add_argument("built", type=Path)
    p_cmp.add_argument("reference", type=Path)
    p_cmp.add_argument("--atol", type=float, default=0.1, help="million USD")

    p_fetch = sub.add_parser("fetch", help="download tables from the mirror into the cache")
    p_fetch.add_argument("countries", nargs="*", help="ISO3 codes (default: all)")
    p_fetch.add_argument("--release", default=DEFAULT_RELEASE)

    p_pkg = sub.add_parser("package", help="make mirror assets from a build folder")
    p_pkg.add_argument("build_dir", type=Path)
    p_pkg.add_argument("out_dir", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "build":
        releases = load_releases()
        if args.release not in releases:
            raise SystemExit(f"unknown release {args.release!r}; known: {sorted(releases)}")
        release = releases[args.release]
        sources, shas = _resolve_sources(
            release, args.bundle, args.cache or default_cache_root(), args.allow_unpinned
        )
        try:
            manifest = build(
                release, sources, args.out, bundle_shas=shas, years=args.years,
                allow_unpinned=args.allow_unpinned,
            )
        except ValueError as err:
            raise SystemExit(str(err)) from err
        print(
            f"wrote {len(manifest['outputs'])} country tables for years "
            f"{manifest['years'][0]}-{manifest['years'][-1]} to {args.out}"
        )
        return 0

    if args.command == "fetch":
        store = TableStore(args.release)
        try:
            countries = [c.upper() for c in args.countries] or store.countries()
            for country in countries:
                store.get(country, store.years()[0], data_dir=None)
        except TableNotFound as err:
            raise SystemExit(str(err)) from err
        print(f"cached {len(countries)} tables in {store.cache_dir}")
        return 0

    if args.command == "package":
        manifest = package_for_mirror(args.build_dir, args.out_dir)
        sha = file_sha256(args.out_dir / "MANIFEST.json")
        print(
            f"wrote {len(manifest['tables'])} tables to {args.out_dir}. Upload the folder's "
            f"contents to the mirror, then set mirror.manifest_sha256: {sha} in releases.yaml"
        )
        return 0

    problems = compare(args.built, args.reference, atol=args.atol)
    for problem in problems:
        print(problem)
    print("OK" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
