"""Build the historical referee-assignment dataset from Basketball-Reference.

Two stages, both resumable and both cached to disk:

1. **index** -- monthly schedule pages give every game's box score address plus
   its date and both team codes. This is what makes the join to ``nba_game_id``
   deterministic: the key is the Eastern date and both canonical team codes,
   never a fuzzy name match.
2. **boxscores** -- one page per game, from which the officiating crew is read.

The second stage is long by construction. Basketball-Reference asks for three
seconds between requests and this honours that, so a full seven-season backfill
is measured in hours, not minutes. It is safe to interrupt: cached pages are
never refetched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.referees.bbref_source import (
    SCHEDULE_MONTHS,
    boxscore_url,
    parse_officials,
    parse_schedule_page,
    schedule_url,
)
from nba_prediction_market.referees.diagnostics import team_referee_support
from nba_prediction_market.referees.fetch import PacedCache, append_log, new_client
from nba_prediction_market.referees.identity import build_registry

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
GAMES_FILE = "nba_regular_season_games_2006_26.parquet"
OUTPUT_FILE = "nba_referee_assignments_2019_26.parquet"
INDEX_FILE = "bbref_schedule_index.json"

SEASONS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

#: Fetch order. Lower runs first: the four development seasons, then the
#: 2025-26 benchmark, then the two earlier seasons that only warm up referee
#: state. Nothing about the modelling depends on this order.
SEASON_FETCH_PRIORITY: dict[int, int] = {
    2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 1, 2019: 2, 2020: 2,
}

#: Mapping quality, recorded per row rather than assumed.
MAPPED = "mapped"
NO_BOXSCORE = "no_boxscore_page"
NO_OFFICIALS_BLOCK = "no_officials_block"
EMPTY_CREW = "empty_crew"


def referee_root(settings: Settings) -> Path:
    return settings.paths.root / "raw" / "referees"


def build_index(settings: Settings, seasons: tuple[int, ...]) -> list[dict[str, Any]]:
    """Every Basketball-Reference game address for the seasons in scope."""
    root = referee_root(settings)
    cache = PacedCache(root / "bbref" / "schedule")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with new_client() as client:
        for season in seasons:
            for month in SCHEDULE_MONTHS:
                key = f"{season}/{month}"
                text, meta = cache.get(key, schedule_url(season, month), client)
                if text is None:
                    # A month a season never reached is a 404, not a failure.
                    if meta.get("http_status") != 404:
                        failures.append(meta)
                    continue
                for game in parse_schedule_page(text):
                    rows.append({
                        "season": season,
                        "boxscore_path": game.path,
                        "game_date_et": game.game_date.isoformat(),
                        "home_team": game.home_team,
                        "away_team": game.away_team,
                        "home_bbref": game.home_bbref,
                        "away_bbref": game.away_bbref,
                    })
            logger.info(
                "indexed season %s: %d games", season,
                sum(1 for r in rows if r["season"] == season),
            )
    (root / INDEX_FILE).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    append_log(root / "index_failures.jsonl", failures)
    logger.info("index complete: %d games, cache %s", len(rows), cache.stats.to_dict())
    return rows


def load_index(settings: Settings) -> list[dict[str, Any]]:
    path = referee_root(settings) / INDEX_FILE
    if not path.is_file():
        raise ConfigError(f"missing {path}; run with --stage index first")
    return json.loads(path.read_text(encoding="utf-8"))


def trusted_games(settings: Settings, seasons: tuple[int, ...]) -> pd.DataFrame:
    """The project's own games, keyed for joining to Basketball-Reference."""
    path = settings.paths.processed / GAMES_FILE
    if not path.is_file():
        raise ConfigError(f"missing {path}")
    games = pd.read_parquet(path)
    games = games[games["season"].isin(seasons)].copy()
    tipoff = pd.to_datetime(games["game_datetime_utc"], utc=True)
    games["game_date_et"] = (
        tipoff.dt.tz_convert(EASTERN).dt.date.astype(str)
    )
    games["join_key"] = (
        games["game_date_et"] + "|" + games["away_team"] + "|" + games["home_team"]
    )
    return games


def fetch_officials(
    settings: Settings,
    index: list[dict[str, Any]],
    *,
    limit: int | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Read the crew off each box score page, caching every page.

    ``offline`` parses only what is already cached and fetches nothing. That
    makes the dataset rebuildable after a parser change without touching the
    network, and lets a partially complete backfill be assembled and inspected
    while it is still running.
    """
    root = referee_root(settings)
    cache = PacedCache(root / "bbref" / "boxscores")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    todo = index if limit is None else index[:limit]

    with new_client() as client:
        for i, entry in enumerate(todo):
            key = entry["boxscore_path"].removeprefix("/boxscores/").removesuffix(".html")
            key = f"{entry['season']}/{key}"
            if offline and not cache.has(key):
                continue
            text, meta = cache.get(key, boxscore_url(entry["boxscore_path"]), client)
            if text is None:
                failures.append({**meta, **entry})
                results.append({**entry, "mapping_quality": NO_BOXSCORE,
                                "officials": [], "crew_size": 0})
                continue
            parsed = parse_officials(text)
            quality = MAPPED
            if not parsed.found_block:
                quality = NO_OFFICIALS_BLOCK
            elif parsed.crew_size == 0:
                quality = EMPTY_CREW
            results.append({
                **entry,
                "mapping_quality": quality,
                "officials": [
                    {"referee_slug": s, "referee_name": n} for s, n in parsed.officials
                ],
                "crew_size": parsed.crew_size,
                "first_observed_at_utc": meta.get("first_observed_at_utc"),
                "page_source": meta.get("source"),
            })
            if (i + 1) % 250 == 0:
                logger.info(
                    "%d/%d box scores (%s)", i + 1, len(todo), cache.stats.to_dict()
                )
                _write_partial(settings, results)

    append_log(root / "boxscore_failures.jsonl", failures)
    logger.info("box scores complete: %s", cache.stats.to_dict())
    return results


def _write_partial(settings: Settings, results: list[dict[str, Any]]) -> None:
    path = referee_root(settings) / "bbref_officials_partial.json"
    path.write_text(json.dumps(results, default=str), encoding="utf-8")


def assemble(
    settings: Settings,
    results: list[dict[str, Any]],
    games: pd.DataFrame,
    index: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join crews to trusted game ids and report exactly what did not map.

    Unmatched games are split into two very different cases. A game the source
    has no entry for is a genuine gap in coverage. A game the source *does*
    list but whose page was not retrieved is a transient failure, and re-running
    the stage fixes it because cached pages are skipped. Reporting them as one
    number would let a handful of timeouts masquerade as missing history.
    """
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        key = f"{row['game_date_et']}|{row['away_team']}|{row['home_team']}"
        by_key.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    ambiguous: list[str] = []
    unmatched: list[str] = []

    for game in games.itertuples():
        candidates = by_key.get(game.join_key, [])
        if not candidates:
            unmatched.append(str(game.nba_game_id))
            continue
        if len(candidates) > 1:
            # Two source rows for one (date, away, home) triple. Never resolved
            # arbitrarily: attaching the wrong crew is silent corruption.
            ambiguous.append(str(game.nba_game_id))
            continue
        found = candidates[0]
        officials = found["officials"]
        rows.append({
            "nba_game_id": game.nba_game_id,
            "season": game.season,
            "game_datetime_utc": game.game_datetime_utc,
            "game_date_et": game.game_date_et,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "crew_size": found["crew_size"],
            "referee_slugs": [o["referee_slug"] for o in officials],
            "referee_names": [o["referee_name"] for o in officials],
            "mapping_quality": found["mapping_quality"],
            "source": "basketball_reference",
            "source_game_identifier": found["boxscore_path"],
            "first_observed_at_utc": found.get("first_observed_at_utc"),
        })

    frame = pd.DataFrame(rows)
    indexed_keys = {
        f"{r['game_date_et']}|{r['away_team']}|{r['home_team']}"
        for r in (index or [])
    }
    unmatched_set = set(unmatched)
    retryable = [
        str(g.nba_game_id) for g in games.itertuples()
        if str(g.nba_game_id) in unmatched_set and g.join_key in indexed_keys
    ]
    absent = sorted(unmatched_set - set(retryable))
    report = {
        "generated_at_utc": utc_now().isoformat(),
        "expected_games": len(games),
        "source_rows": len(results),
        "matched_games": len(frame),
        "unmatched_games": len(unmatched),
        # Listed by the source but not retrieved: re-run the stage, which skips
        # everything already cached and so retries only these.
        "unretrieved_but_listed": len(retryable),
        "unretrieved_examples": retryable[:20],
        # Genuinely absent from the source.
        "absent_from_source": len(absent),
        "absent_examples": absent[:20],
        "ambiguous_games": len(ambiguous),
        "ambiguous_examples": ambiguous[:20],
        "repair_command": (
            "python -m nba_prediction_market.pipelines.build_referee_assignments "
            "--stage boxscores"
        ) if retryable else None,
    }
    if not frame.empty:
        registry = build_registry([
            {
                "nba_game_id": row.nba_game_id,
                "referee_slugs": list(row.referee_slugs),
                "referee_names": list(row.referee_names),
            }
            for row in frame.itertuples()
        ])
        report["identity"] = registry.summary()
        report["team_referee_support"] = team_referee_support(frame)
        report["by_quality"] = (
            frame["mapping_quality"].value_counts().to_dict()
        )
        report["crew_size_distribution"] = (
            frame["crew_size"].value_counts().sort_index().to_dict()
        )
        report["by_season"] = {
            str(season): {
                "games": len(part),
                "with_crew": int((part["crew_size"] > 0).sum()),
            }
            for season, part in frame.groupby("season")
        }
    return frame, report


def run_pipeline(
    *,
    settings: Settings | None = None,
    stage: str = "all",
    seasons: tuple[int, ...] = SEASONS,
    limit: int | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    referee_root(settings).mkdir(parents=True, exist_ok=True)

    index = (
        build_index(settings, seasons)
        if stage in ("index", "all") and not offline
        else load_index(settings)
    )
    index = [r for r in index if r["season"] in seasons]

    if stage == "index":
        return {"stage": "index", "indexed_games": len(index)}

    games = trusted_games(settings, seasons)
    # Only fetch pages that can actually join. The index includes playoff games,
    # which the trusted regular-season frame does not carry, and spending the
    # crawl budget on pages that cannot be used is the one cost with no upside.
    wanted = set(games["join_key"])
    eligible = [
        r for r in index
        if f"{r['game_date_et']}|{r['away_team']}|{r['home_team']}" in wanted
    ]
    # Development and benchmark seasons first. The fetch order is independent
    # of the chronological order state is later built in, so front-loading the
    # seasons the experiment needs means an interrupted backfill still leaves a
    # usable dataset rather than a warm-up period and nothing to test on.
    eligible.sort(key=lambda r: (SEASON_FETCH_PRIORITY.get(r["season"], 9),
                                 r["game_date_et"]))
    logger.info(
        "%d indexed games, %d join the trusted regular-season frame",
        len(index), len(eligible),
    )
    results = fetch_officials(settings, eligible, limit=limit, offline=offline)
    frame, report = assemble(settings, results, games, index=index)

    written = []
    if not frame.empty:
        out = settings.paths.processed / OUTPUT_FILE
        frame.to_parquet(out, index=False)
        written.append(str(out))
    path = settings.paths.reports / "referee_assignment_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    written.append(str(path))
    report["written_files"] = written
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build referee assignments.")
    parser.add_argument("--stage", choices=["index", "boxscores", "all"], default="all")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated start years. Default: 2019..2025.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many games. For smoke tests.")
    parser.add_argument("--offline", action="store_true",
                        help="Parse cached pages only; fetch nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    seasons = (
        tuple(int(s) for s in args.seasons.split(",")) if args.seasons else SEASONS
    )
    try:
        report = run_pipeline(stage=args.stage, seasons=seasons, limit=args.limit,
                              offline=args.offline)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
