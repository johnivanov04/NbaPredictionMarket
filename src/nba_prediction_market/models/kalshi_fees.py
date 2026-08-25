"""Date-effective Kalshi trading fees.

Fees decide whether a paper edge survives contact with the exchange, so the
schedule is not hardcoded from memory. Each entry below is transcribed from a
dated copy of Kalshi's published fee schedule, retrieved from the Internet
Archive so the version *in force on a historical trade date* can be used rather
than today's.

A date outside every known window raises rather than falling back to the newest
schedule: silently applying the wrong fees would quietly change the answer to
the only question this phase asks.

Two properties matter for the simulation:

* **Fees round up to the next cent on the whole order**, not per contract. The
  per-contract cost therefore falls with order size -- $0.02 at one contract,
  $0.0175 at a hundred, at a 50c price. Reporting an infinitesimal theoretical
  fee would hide that.
* **Maker fees are charged only where Kalshi designates them**, and a resting
  order is not a fill. Maker economics are a sensitivity, never realised P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Any, Final

#: Where each schedule was retrieved from, for the report's provenance trail.
ARCHIVE_TEMPLATE: Final = (
    "https://web.archive.org/web/{timestamp}id_/https://kalshi.com/docs/"
    "kalshi-fee-schedule.pdf"
)


@dataclass(frozen=True)
class FeeSchedule:
    """One published schedule and the window it governs."""

    effective_from: date
    effective_to: date | None
    taker_multiplier: float
    maker_multiplier: float | None
    source_timestamp: str
    stated_effective: str
    notes: str

    @property
    def source_url(self) -> str:
        return ARCHIVE_TEMPLATE.format(timestamp=self.source_timestamp)

    def covers(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_to is None or when <= self.effective_to

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "stated_effective": self.stated_effective,
            "taker_formula": (
                f"fees = round up({self.taker_multiplier} x C x P x (1-P))"
            ),
            "maker_formula": (
                f"fees = round up({self.maker_multiplier} x C x P x (1-P))"
                if self.maker_multiplier is not None else None
            ),
            "rounding": "round up to the next cent, applied to the whole order",
            "product_specific_exception_for_nba": False,
            "source_url": self.source_url,
            "notes": self.notes,
        }


_NO_NBA_EXCEPTION = (
    "Verified on the document itself: zero mentions of NBA, Basketball, Sports "
    "or KXNBAGAME. The only product-specific schedules are S&P 500 and "
    "Nasdaq-100 index products, so NBA game markets take the general formula."
)

#: Schedules covering the 2025-26 NBA regular season. The two published
#: versions carry *identical* general formulas; the February revision changed
#: other parts of the document.
FEE_SCHEDULES: Final[tuple[FeeSchedule, ...]] = (
    FeeSchedule(
        effective_from=date(2025, 10, 1),
        effective_to=date(2026, 2, 4),
        taker_multiplier=0.07,
        maker_multiplier=0.0175,
        source_timestamp="20251008232930",
        stated_effective="Last updated and effective: Oct 1, 2025",
        notes=_NO_NBA_EXCEPTION,
    ),
    FeeSchedule(
        effective_from=date(2026, 2, 5),
        effective_to=date(2026, 12, 31),
        taker_multiplier=0.07,
        maker_multiplier=0.0175,
        source_timestamp="20260214014036",
        stated_effective="Last updated and effective: Feb 5, 2026",
        notes=_NO_NBA_EXCEPTION,
    ),
)


def _round_up_cents(multiplier: float, price: float, contracts: int) -> float:
    """``round up(multiplier x C x P x (1-P))`` to the next cent, exactly.

    Done in decimal rather than binary floating point. At 100 contracts and a
    50c price the exact product is $1.75, but in binary it lands a hair above
    and ceilings to $1.76 -- a systematic overstatement of the fee, always in
    the direction that makes a strategy look worse than it is.
    """
    exact = (
        Decimal(str(multiplier))
        * Decimal(contracts)
        * Decimal(str(price))
        * (Decimal(1) - Decimal(str(price)))
    )
    cents = (exact * Decimal(100)).quantize(Decimal(1), rounding=ROUND_CEILING)
    return float(cents) / 100.0


class FeeScheduleUnavailableError(RuntimeError):
    """Raised when no published schedule covers a trade date."""


def schedule_for(when: date) -> FeeSchedule:
    """The schedule in force on ``when``.

    Raises rather than guessing. A trade date outside every documented window
    means the fee is unknown, and an unknown fee cannot be quietly replaced by
    the latest one without changing what the simulation is measuring.
    """
    for schedule in FEE_SCHEDULES:
        if schedule.covers(when):
            return schedule
    covered = ", ".join(
        f"{s.effective_from}..{s.effective_to or 'open'}" for s in FEE_SCHEDULES
    )
    raise FeeScheduleUnavailableError(
        f"no published Kalshi fee schedule covers {when}; documented windows: {covered}"
    )


def taker_fee(price: float, contracts: int, when: date) -> float:
    """Total taker fee in dollars for ``contracts`` bought at ``price``.

    ``round up`` is applied once to the whole order, exactly as published, which
    is why this is not ``contracts * per_contract_fee``.
    """
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"price must be a probability in dollars, got {price}")
    return _round_up_cents(schedule_for(when).taker_multiplier, price, contracts)


def maker_fee(price: float, contracts: int, when: date) -> float | None:
    """Total maker fee, or ``None`` where the schedule defines none."""
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    multiplier = schedule_for(when).maker_multiplier
    if multiplier is None:
        return None
    return _round_up_cents(multiplier, price, contracts)


def taker_fee_per_contract(price: float, contracts: int, when: date) -> float:
    """Effective per-contract taker fee at a given order size."""
    return taker_fee(price, contracts, when) / contracts
