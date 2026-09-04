"""Referee identity, resolved on a stable key rather than on a name.

Basketball-Reference gives every official a slug (``cutleke99r``) that is
stable across seasons and across spelling changes. That slug is the identity
here; the display name is carried for reporting only.

This ordering matters. Names are not unique (the league has fielded more than
one Williams), and they change -- so keying on a name would silently merge two
officials or split one. Nothing in this module ever fuzzy-matches: a name that
does not resolve is reported, never guessed into the nearest slug.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile("[.'`\u2019]")  # includes the curly apostrophe
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_name(name: str) -> str:
    """A comparable form of a referee's display name.

    Folds accents, strips punctuation and generational suffixes, and collapses
    whitespace. Used to *detect* disagreement between sources, never to join:
    two officials who normalize alike are reported as a collision, because
    resolving them by similarity is exactly the silent error to avoid.
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCTUATION.sub("", folded).lower()
    parts = [p for p in _WHITESPACE.split(folded.strip()) if p]
    while parts and parts[-1].rstrip(".") in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


@dataclass(frozen=True)
class Referee:
    """One official, keyed by source slug."""

    referee_id: str
    display_name: str
    normalized_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "referee_id": self.referee_id,
            "display_name": self.display_name,
            "normalized_name": self.normalized_name,
        }


@dataclass
class RefereeRegistry:
    """Every official seen, with the disagreements found while building it."""

    by_id: dict[str, Referee] = field(default_factory=dict)
    #: slug -> every distinct display name seen for it. More than one is
    #: normal (a rename); it is recorded rather than resolved.
    aliases: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    #: normalized name -> every slug carrying it. More than one means two
    #: officials share a name, which is why the slug is the identity.
    name_collisions: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add(self, referee_id: str, display_name: str) -> Referee:
        if not referee_id:
            raise ValueError("referee_id is required; a name is not an identity")
        normalized = normalize_name(display_name)
        self.aliases[referee_id].add(display_name)
        self.name_collisions[normalized].add(referee_id)
        existing = self.by_id.get(referee_id)
        if existing is not None:
            return existing
        referee = Referee(referee_id, display_name, normalized)
        self.by_id[referee_id] = referee
        return referee

    def resolve(self, referee_id: str) -> Referee | None:
        """Never guesses. An unknown slug is unknown."""
        return self.by_id.get(referee_id)

    @property
    def renamed(self) -> dict[str, list[str]]:
        return {
            rid: sorted(names)
            for rid, names in self.aliases.items() if len(names) > 1
        }

    @property
    def shared_names(self) -> dict[str, list[str]]:
        return {
            name: sorted(ids)
            for name, ids in self.name_collisions.items() if len(ids) > 1
        }

    def summary(self) -> dict[str, Any]:
        return {
            "distinct_referees": len(self.by_id),
            "referees_with_multiple_display_names": len(self.renamed),
            "normalized_names_shared_by_multiple_ids": len(self.shared_names),
            "rename_examples": dict(list(self.renamed.items())[:10]),
            "shared_name_examples": dict(list(self.shared_names.items())[:10]),
        }


def build_registry(rows: list[dict[str, Any]]) -> RefereeRegistry:
    """Registry from assignment rows carrying ``referee_slugs``/``referee_names``."""
    registry = RefereeRegistry()
    for row in rows:
        slugs = row.get("referee_slugs") or []
        names = row.get("referee_names") or []
        if len(slugs) != len(names):
            raise ValueError(
                f"crew for game {row.get('nba_game_id')} has {len(slugs)} slugs "
                f"and {len(names)} names; refusing to pair them by position"
            )
        for slug, name in zip(slugs, names, strict=True):
            registry.add(slug, name)
    return registry
