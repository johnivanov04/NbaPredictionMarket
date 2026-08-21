"""Cross-source player identity resolution.

Availability feeds introduce new player namespaces. Nothing is fuzzy-matched:
a player either resolves through a registered identifier or is reported as
unresolved, exactly as team names were handled in Phase 1.

Resolution order, most authoritative first:

1. **NBA reference id** -- the league's own id, when a source supplies it.
2. **A registered cross-source alias** -- an explicit, reviewed mapping.
3. **Exact normalized name within a team** -- the weakest key, and the reason
   ``team_id`` is required for it. Names alone collide (there have been two
   Marcus Williamses); a name plus a team in one season rarely does.

Anything else is unresolved and reported.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final

# Strips both apostrophe forms; the typographic one appears in real payloads.
_PUNCTUATION = re.compile(r"[.'’]")  # noqa: RUF001
_WHITESPACE = re.compile(r"\s+")
_SUFFIXES: Final[frozenset[str]] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalize_name(value: str | None) -> str:
    """Casefold to a comparable key, handling both name conventions.

    The NBA injury report writes ``"Porter Jr., Michael"`` while BALLDONTLIE
    writes ``"Michael Porter Jr."``. Both must reduce to the same key, so a
    comma is treated as a surname/given-name separator and generational suffixes
    are removed wherever they appear -- not only at the end, which was the
    difference that made the two conventions disagree.
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = _PUNCTUATION.sub("", stripped)

    if "," in cleaned:
        surname, _, given = cleaned.partition(",")
        cleaned = f"{given} {surname}"

    tokens = [
        token
        for token in _WHITESPACE.sub(" ", cleaned).strip().casefold().split()
        if token and token not in _SUFFIXES
    ]
    return " ".join(tokens)


@dataclass(frozen=True)
class ResolvedPlayer:
    """A resolution outcome. Only ``resolved`` carries an id."""

    raw_name: str
    status: str
    balldontlie_player_id: Any | None = None
    nba_reference_id: Any | None = None
    method: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "resolved"


@dataclass
class PlayerRegistry:
    """Known players, keyed by every identifier we trust."""

    by_nba_reference: dict[Any, Any] = field(default_factory=dict)
    by_alias: dict[str, Any] = field(default_factory=dict)
    by_team_and_name: dict[tuple[Any, str], Any] = field(default_factory=dict)
    names: dict[Any, str] = field(default_factory=dict)

    def register(
        self,
        balldontlie_player_id: Any,
        full_name: str,
        *,
        team_id: Any | None = None,
        nba_reference_id: Any | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register one player. Conflicting registrations raise rather than win."""
        self.names[balldontlie_player_id] = full_name
        if nba_reference_id is not None:
            existing = self.by_nba_reference.get(nba_reference_id)
            if existing is not None and existing != balldontlie_player_id:
                raise ValueError(
                    f"NBA reference id {nba_reference_id} already maps to {existing}; "
                    f"refusing to remap to {balldontlie_player_id}"
                )
            self.by_nba_reference[nba_reference_id] = balldontlie_player_id
        if team_id is not None:
            self.by_team_and_name[(team_id, normalize_name(full_name))] = (
                balldontlie_player_id
            )
        for alias in aliases:
            key = normalize_name(alias)
            existing = self.by_alias.get(key)
            if existing is not None and existing != balldontlie_player_id:
                raise ValueError(
                    f"alias {alias!r} is claimed by both {existing} and "
                    f"{balldontlie_player_id}; resolve it explicitly"
                )
            self.by_alias[key] = balldontlie_player_id

    def resolve(
        self,
        player_name: str | None,
        *,
        nba_reference_id: Any | None = None,
        team_id: Any | None = None,
    ) -> ResolvedPlayer:
        """Resolve one source record to a known player, or report why not."""
        raw = player_name or ""
        if nba_reference_id is not None:
            mapped = self.by_nba_reference.get(nba_reference_id)
            if mapped is not None:
                return ResolvedPlayer(
                    raw, "resolved", mapped, nba_reference_id, "nba_reference_id"
                )
            return ResolvedPlayer(
                raw, "unresolved", None, nba_reference_id, None,
                f"NBA reference id {nba_reference_id} is not registered",
            )

        key = normalize_name(raw)
        if not key:
            return ResolvedPlayer(raw, "unresolved", reason="blank player name")
        alias = self.by_alias.get(key)
        if alias is not None:
            return ResolvedPlayer(raw, "resolved", alias, None, "alias")
        if team_id is not None:
            mapped = self.by_team_and_name.get((team_id, key))
            if mapped is not None:
                return ResolvedPlayer(raw, "resolved", mapped, None, "team_and_name")
            return ResolvedPlayer(
                raw, "unresolved", reason=f"no player {raw!r} registered for team {team_id}"
            )
        return ResolvedPlayer(
            raw, "unresolved",
            reason="name-only resolution requires a team_id; names alone collide",
        )


def unresolved_report(results: list[ResolvedPlayer]) -> dict[str, Any]:
    """Summarise resolution outcomes for the audit."""
    unresolved = [r for r in results if not r.ok]
    return {
        "total": len(results),
        "resolved": len(results) - len(unresolved),
        "unresolved": len(unresolved),
        "resolution_methods": {
            method: sum(1 for r in results if r.method == method)
            for method in ("nba_reference_id", "alias", "team_and_name")
        },
        "unresolved_examples": [
            {"name": r.raw_name, "reason": r.reason} for r in unresolved[:20]
        ],
    }
