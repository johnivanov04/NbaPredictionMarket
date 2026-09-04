"""Basketball-Reference as the historical source of game officials.

Chosen after auditing the alternatives, and chosen with known costs:

* **stats.nba.com is the better source and is not usable here.** Its
  ``boxscoresummaryv2`` endpoint returns an ``Officials`` result set carrying
  a stable ``OFFICIAL_ID`` *and* an order that appears positional. It served
  one request from this environment and then hard-blocked the address: five
  consecutive retries timed out after a cooldown, and it did not recover. A
  backfill of ~8,300 games cannot be built on it from here.
* **Basketball-Reference lists officials alphabetically**, so it cannot supply
  the crew-chief / referee / umpire *positions*. Verified directly: the
  2023-10-24 DEN box score lists Cutler, Twardoski, Williams, which is
  alphabetical, while the NBA's own ordering for a different game (Forte,
  Barnaky, Mehta) is not. Position-specific features are therefore not
  buildable from this source, and this module does not pretend otherwise --
  it exposes officials as an unordered crew.
* **Per-referee pages carry only season aggregates** (``raw_``, ``relative_``,
  ``rs_home_vs_visitor``), which are end-of-season summaries. Using them
  retrospectively would leak, so they are not used.

Access is paced at Basketball-Reference's own ``Crawl-delay: 3``. Nothing here
touches a path their robots.txt disallows: ``/boxscores/`` and ``/leagues/``
are both permitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

BASE_URL: Final = "https://www.basketball-reference.com"

#: Basketball-Reference's own ``Crawl-delay``, with a margin. Not tunable: it
#: is their stated limit, not a performance knob.
CRAWL_DELAY_SECONDS: Final = 3.2

#: Months a season's schedule pages can appear under. October through June
#: covers every regular season in scope plus the 2019-20 and 2020-21 bubbles,
#: whose games ran into July/August and December respectively.
SCHEDULE_MONTHS: Final[tuple[str, ...]] = (
    "october", "november", "december", "january", "february",
    "march", "april", "may", "june", "july", "august", "september",
)

#: Basketball-Reference team codes that differ from this project's canonical
#: abbreviations. Explicit and total: a code absent from here is assumed to
#: match, and an unknown code is reported rather than guessed.
BBREF_TO_CANONICAL: Final[dict[str, str]] = {
    "PHO": "PHX",
    "BRK": "BKN",
    "CHO": "CHA",
    "NJN": "BKN",
    "NOH": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
    "CHH": "NOP",
}

_BOXSCORE_HREF = re.compile(r'href="(/boxscores/(\d{8})0([A-Z]{3})\.html)"')
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TEAM = re.compile(
    r'data-stat="(visitor_team_name|home_team_name)"[^>]*>'
    r'<a href="/teams/([A-Z]{3})/'
)
#: The officials block. Anchored on the literal label so a page without one is
#: reported missing rather than silently matching some other link list.
_OFFICIALS_BLOCK = re.compile(
    r"<strong>\s*Officials:\s*(?:&nbsp;)?\s*</strong>(.*?)</div>", re.S
)
_OFFICIAL_LINK = re.compile(r"href='/referees/([a-z0-9]+)\.html'>([^<]+)</a>")


def canonical_team(bbref_code: str) -> str:
    """This project's abbreviation for a Basketball-Reference team code."""
    return BBREF_TO_CANONICAL.get(bbref_code, bbref_code)


def schedule_url(season: int, month: str) -> str:
    """Monthly schedule page. ``season`` is the start year, as elsewhere."""
    return f"{BASE_URL}/leagues/NBA_{season + 1}_games-{month}.html"


def boxscore_url(path: str) -> str:
    return f"{BASE_URL}{path}"


@dataclass(frozen=True)
class ScheduledBoxscore:
    """One game located on a schedule page, with its box score address."""

    path: str
    game_date: date
    home_bbref: str
    away_bbref: str

    @property
    def home_team(self) -> str:
        return canonical_team(self.home_bbref)

    @property
    def away_team(self) -> str:
        return canonical_team(self.away_bbref)

    @property
    def key(self) -> tuple[str, str, str]:
        """Deterministic join key: ET date plus both canonical team codes."""
        return (self.game_date.isoformat(), self.away_team, self.home_team)


def parse_schedule_page(html: str) -> list[ScheduledBoxscore]:
    """Every game on a monthly schedule page.

    A row without a box score link is skipped: on a current-season page those
    are games not yet played, which is a normal state rather than an error.
    """
    found: list[ScheduledBoxscore] = []
    for row in _ROW.findall(html):
        link = _BOXSCORE_HREF.search(row)
        if link is None:
            continue
        teams = dict(
            (stat, code) for stat, code in _TEAM.findall(row)
        )
        home = teams.get("home_team_name")
        away = teams.get("visitor_team_name")
        if home is None or away is None:
            continue
        stamp = link.group(2)
        found.append(
            ScheduledBoxscore(
                path=link.group(1),
                game_date=date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])),
                home_bbref=home,
                away_bbref=away,
            )
        )
    return found


@dataclass(frozen=True)
class ParsedOfficials:
    """The officiating crew read off one box score page.

    ``officials`` is an *unordered* crew. Basketball-Reference sorts the names
    alphabetically, so position cannot be recovered and is not claimed.
    """

    officials: tuple[tuple[str, str], ...]  # (slug, display name)
    found_block: bool

    @property
    def crew_size(self) -> int:
        return len(self.officials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "officials": [
                {"referee_slug": s, "referee_name": n} for s, n in self.officials
            ],
            "crew_size": self.crew_size,
            "found_block": self.found_block,
        }


def parse_officials(html: str) -> ParsedOfficials:
    """Officials from a box score page.

    Distinguishes "the page has no officials block" from "the block is there
    and empty". The first is a page we failed to understand; the second is a
    game the source genuinely has no crew for. Collapsing them would let a
    parser regression masquerade as missing data.
    """
    block = _OFFICIALS_BLOCK.search(html)
    if block is None:
        return ParsedOfficials(officials=(), found_block=False)
    pairs = [
        (slug, re.sub(r"\s+", " ", name).strip())
        for slug, name in _OFFICIAL_LINK.findall(block.group(1))
    ]
    return ParsedOfficials(officials=tuple(pairs), found_block=True)
