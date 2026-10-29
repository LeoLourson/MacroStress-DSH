"""``python -m mst.corpus``: build the scenario corpus of corpus/README.md.

::

  python -m mst.corpus check --corpus DIR
  python -m mst.corpus fetch [ID ...] [--force] --corpus DIR
  python -m mst.corpus discover URL [--match RE]
  python -m mst.corpus add KEY FILE [--url URL] --corpus DIR
  python -m mst.corpus convert [ID ...] --corpus DIR
  python -m mst.corpus series [SERIES ...] --corpus DIR --store DIR
  python -m mst.corpus episodes [ID ...] [--dry-run] --corpus DIR --store DIR
  python -m mst.corpus index --corpus DIR
  python -m mst.corpus search QUERY --as-of DAY --corpus DIR
  python -m mst.corpus eval --corpus DIR
  python -m mst.corpus all --corpus DIR --store DIR

``--corpus`` is the corpus root (``workspace/corpus``): the manifest, ``series.toml``, the
Markdown texts, and ``raw/``, ``text/``, ``series/``, ``index/``, ``downloads.json`` that the
commands write. ``--store`` is the monthly series store (``workspace/series``) the episode cards
fall back to when a daily series is not downloaded.

Requests go only to the addresses in the manifest and to the series sources (Bank of Russia,
Moscow Exchange, FRED with FRED_API_KEY, BIS). A run that fails on some files exits with 1 and
names them; the files that came through are kept.
"""

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from . import manifest as manifest_module
from .convert import convert
from .episodes import measure, write_card, written
from .fetch import add, client, discover, fetch, load_downloads, save_downloads
from .index import Index, build, questions, recall, write, write_state, write_vectors
from .manifest import documents
from .series import Cache, fetch_series, load_catalog


def check(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    catalog = load_catalog(root / "series.toml")
    downloads = load_downloads(root / "downloads.json")
    docs = [d for s in manifest.sources for d in documents(s)]
    unspecified = [d.key for d in docs if d.spec is None]
    fetched = [d.key for d in docs if d.key in downloads]
    converted = [d.key for d in docs if (root / d.path).exists()]
    print(f"sources: {len(manifest.sources)}, files: {len(docs)}")
    print(
        f"  with an address: {len(docs) - len(unspecified)}, downloaded: {len(fetched)}, "
        f"in Markdown: {len(converted)}"
    )
    print(f"  status: {dict(Counter(s.status for s in manifest.sources))}")
    if unspecified:
        print(f"  no address yet: {', '.join(unspecified)}")
    cards = [e for e in manifest.episodes if (root / e.path).exists()]
    chronologies = [e for e in cards if written((root / e.path).read_text(encoding="utf-8"))]
    print(
        f"episodes: {len(manifest.episodes)}, cards: {len(cards)}, "
        f"chronologies written: {len(chronologies)}"
    )
    print(f"  status: {dict(Counter(e.status for e in manifest.episodes))}")
    print(f"series: {len(catalog.series)}")
    return 0


def run_fetch(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    with client() as http:
        outcome = fetch(
            manifest,
            http,
            set(args.ids) or None,
            root=root / "raw",
            downloads_path=root / "downloads.json",
            force=args.force,
        )
    print(
        f"fetched {len(outcome.fetched)}, already there {len(outcome.kept)}, "
        f"no address {len(outcome.unspecified)}"
    )
    for key, reason in outcome.failed.items():
        print(f"{key}: failed, {reason}", file=sys.stderr)
    return 1 if outcome.failed else 0


def run_discover(args) -> int:
    with client() as http:
        for link in discover(http, args.url, args.match):
            print(f"{link.url}\t{link.text}")
    return 0


def run_add(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    found = [d for s in manifest.sources for d in documents(s) if d.key == args.key]
    if not found:
        print(f"no document {args.key} in the manifest", file=sys.stderr)
        return 2
    document = found[0]
    url = args.url or (document.spec.url or document.spec.page if document.spec else None)
    if url is None:
        print(f"{args.key} has no address in the manifest: give --url", file=sys.stderr)
        return 2
    records = load_downloads(root / "downloads.json")
    records[document.key] = add(document, args.file, url, root / "raw")
    save_downloads(records, root / "downloads.json")
    print(f"{document.key}: {document.raw_path(root / 'raw')}")
    return 0


def run_convert(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    downloads = load_downloads(root / "downloads.json")
    failed, changed = {}, False
    for i, source in enumerate(manifest.sources):
        if args.ids and source.id not in args.ids:
            continue
        done = 0
        for document in documents(source):
            record = downloads.get(document.key)
            if record is None or not document.raw_path(root / "raw").exists():
                continue
            try:
                result = convert(
                    document,
                    record,
                    raw_root=root / "raw",
                    corpus_root=root,
                    private_root=root / "text",
                )
            except Exception as error:  # noqa: BLE001 - one broken file does not stop the rest
                failed[document.key] = f"{type(error).__name__}: {error}"
                continue
            done += 1
            empty = (
                f", {result.empty_pages} of {result.pages} pages without text"
                if (result.empty_pages)
                else ""
            )
            print(f"{document.key}: {result.path.relative_to(root)}{empty}")
            if result.pages and result.empty_pages > result.pages / 2:
                print(f"{document.key}: mostly a scan, needs OCR", file=sys.stderr)
        if done and source.status == "planned" and source.text_policy == "full":
            manifest.sources[i] = source.model_copy(update={"status": "raw"})
            changed = True
    if changed:
        manifest_module.save(manifest, root / "manifest.json")
    for key, reason in failed.items():
        print(f"{key}: failed, {reason}", file=sys.stderr)
    return 1 if failed else 0


def run_series(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    catalog = load_catalog(root / "series.toml")
    if args.series:
        ids = args.series
    else:
        ids = sorted(
            {
                sid
                for e in manifest.episodes
                for f in e.factors
                for sid in catalog.candidates(e.id, e.country, f)
            }
        )
        ids += sorted({catalog.series[s].minus for s in ids if catalog.series[s].minus})
    unknown = sorted(set(ids) - set(catalog.series))
    if unknown:
        print(f"unknown series {', '.join(unknown)}", file=sys.stderr)
        return 2
    with client() as http:
        failed = fetch_series(catalog, Cache(root / "series", args.store), http, ids)
    print(f"series: {len(ids) - len(failed)} of {len(ids)}")
    for series_id, reason in failed.items():
        print(f"{series_id}: failed, {reason}", file=sys.stderr)
    return 1 if failed else 0


def run_episodes(args) -> int:
    root = args.corpus
    manifest = manifest_module.load(root / "manifest.json")
    catalog, cache = load_catalog(root / "series.toml"), Cache(root / "series", args.store)
    changed = False
    for i, episode in enumerate(manifest.episodes):
        if args.ids and episode.id not in args.ids:
            continue
        if args.dry_run:
            measured = measure(episode, catalog, cache)
            print(
                f"{episode.id}: {len(measured.movements)} measured, {len(measured.unmeasured)} not"
            )
            continue
        path, measured = write_card(episode, catalog, cache, root)
        print(f"{episode.id}: {len(measured.movements)} of {len(episode.factors)} factors measured")
        if episode.status == "planned":
            manifest.episodes[i] = episode.model_copy(update={"status": "draft"})
            changed = True
    if changed:
        manifest_module.save(manifest, root / "manifest.json")
    return 0


def run_index(args) -> int:
    root = args.corpus
    built = build(manifest_module.load(root / "manifest.json"), root)
    path = write(built.chunks, root / "index")
    print(f"{len(built.chunks)} chunks in {path}")
    computed = write_vectors(built.chunks, root / "index")
    print(f"vectors: {computed} computed, {len(built.chunks) - computed} reused")
    snapshot = write_state(built.chunks, root / "index", date.today())
    print(f"index state: as of {snapshot['as_of']}, {snapshot['chunks']} chunks")
    if built.skipped:
        print(f"not converted yet: {', '.join(built.skipped)}")
    for error in built.errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if built.errors else 0


def run_search(args) -> int:
    index = Index.load(args.corpus / "index")
    factors = args.factor or None
    for chunk in index.search(args.query, args.as_of, args.kind, args.region, factors, args.k):
        pages = f" p. {', '.join(chunk.pages)}" if chunk.pages else ""
        print(f"{chunk.chunk_id}{pages} | {' / '.join(chunk.heading_path)}")
        print("   " + " ".join(chunk.text.split())[:300])
    return 0


def run_eval(args) -> int:
    items = questions((args.corpus / "eval" / "questions.md").read_text(encoding="utf-8"))
    value, unmarked = recall(Index.load(args.corpus / "index"), items)
    print(f"questions: {len(items)}, marked up: {len(items) - len(unmarked)}")
    print("recall@5: " + ("none, nothing marked up" if value is None else f"{value:.3f}"))
    return 0


def run_all(args) -> int:
    worst = 0
    for step in (run_fetch, run_convert, run_series, run_episodes, run_index):
        worst = max(
            worst,
            step(
                argparse.Namespace(
                    corpus=args.corpus,
                    store=args.store,
                    ids=[],
                    force=False,
                    series=[],
                    dry_run=False,
                )
            ),
        )
    return worst


def main(argv: list[str] | None = None) -> int:
    # Russian text reaches a Windows console or pipe intact, not in its ANSI code page.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m mst.corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = argparse.ArgumentParser(add_help=False)
    corpus.add_argument("--corpus", type=Path, required=True, help="the corpus root")
    store = argparse.ArgumentParser(add_help=False)
    store.add_argument("--store", type=Path, required=True, help="the monthly series store")
    commands.add_parser("check", parents=[corpus]).set_defaults(run=check)
    command = commands.add_parser("fetch", parents=[corpus])
    command.add_argument("ids", nargs="*")
    command.add_argument("--force", action="store_true")
    command.set_defaults(run=run_fetch)
    command = commands.add_parser("discover")
    command.add_argument("url")
    command.add_argument("--match")
    command.set_defaults(run=run_discover)
    command = commands.add_parser("add", parents=[corpus])
    command.add_argument("key")
    command.add_argument("file", type=Path)
    command.add_argument("--url")
    command.set_defaults(run=run_add)
    command = commands.add_parser("convert", parents=[corpus])
    command.add_argument("ids", nargs="*")
    command.set_defaults(run=run_convert)
    command = commands.add_parser("series", parents=[corpus, store])
    command.add_argument("series", nargs="*")
    command.set_defaults(run=run_series)
    command = commands.add_parser("episodes", parents=[corpus, store])
    command.add_argument("ids", nargs="*")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(run=run_episodes)
    commands.add_parser("index", parents=[corpus]).set_defaults(run=run_index)
    command = commands.add_parser("search", parents=[corpus])
    command.add_argument("query")
    command.add_argument("--as-of", type=date.fromisoformat, required=True)
    command.add_argument("--kind")
    command.add_argument("--region")
    command.add_argument("--factor", action="append")
    command.add_argument("-k", type=int, default=8)
    command.set_defaults(run=run_search)
    commands.add_parser("eval", parents=[corpus]).set_defaults(run=run_eval)
    commands.add_parser("all", parents=[corpus, store]).set_defaults(run=run_all)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
