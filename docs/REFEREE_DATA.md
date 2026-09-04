# Referee data: sources, limits, and what can be built from them

Phase 4A4. This document records the source audit, because the limitations it
found shape every feature downstream and are easy to forget once the parquet
exists.

## Why this family at all

The NBA publishes each day's officiating crews at roughly 09:00 ET, hours
before the earliest prediction anchor (T-6h). Unlike another rearrangement of
past scores, the identity of tonight's crew is genuinely new pregame
information. That is the only reason it is worth testing.

## Sources audited

| source | officials? | stable ids? | positions? | usable here |
| --- | --- | --- | --- | --- |
| stats.nba.com `boxscoresummaryv2` | yes | yes (`OFFICIAL_ID`) | order appears positional | **no — blocked** |
| Basketball-Reference box scores | yes | yes (slug) | **no — alphabetical** | **yes** |
| Basketball-Reference referee pages | aggregates only | yes | no | no — retrospective |
| official.nba.com assignments | yes | no | **yes** | prospective only |

### stats.nba.com — the better source, and unavailable

`boxscoresummaryv2` returns an `Officials` result set with `OFFICIAL_ID`,
first/last name and jersey number, and its ordering appears to be positional
(a sampled game returned Forte, Barnaky, Mehta — not alphabetical). It served
**one** request from this environment and then blocked the address: five
consecutive retries timed out after a cooldown, and it had not recovered
minutes later. `leaguegamelog` timed out repeatedly even before that. A
backfill of ~8,300 games cannot be built on it from here.

### Basketball-Reference — chosen

Box score pages carry the crew as links to per-referee slugs
(`/referees/cutleke99r.html`), which is a stable identity that survives
spelling changes and distinguishes two officials who share a name.

Access rules, from their robots.txt: `Crawl-delay: 3`, and neither
`/boxscores/` nor `/leagues/` is disallowed. The backfill paces at 3.2 s
between request *starts* and caches every page, so a parser change never
refetches and an interrupted run resumes where it stopped. A full seven-season
backfill therefore takes about **7.4 hours** — that is the cost of being
polite, and it is not negotiable downward.

**The critical limitation: officials are listed alphabetically.** Verified on
the 2023-10-24 Denver box score (Cutler, Twardoski, Williams). Name order is
therefore not position order, and crew chief / referee / umpire **cannot be
recovered from this source**. No feature in this phase claims otherwise.

### Basketball-Reference referee pages — rejected

`/referees/<slug>.html` carries `raw_`, `relative_` and `rs_home_vs_visitor`
tables. These are season aggregates — end-of-season summaries. Using them to
describe a mid-season game would score that game against a number that did not
exist yet, which is leakage even though no individual game is misused.

### official.nba.com — prospective only

The published page has exactly the columns the modelling wants: **Game, Crew
Chief, Referee, Umpire, Alternate**. Two constraints:

* it shows the current day only, and its date-filtered form carries a query
  string, which `official.nba.com/robots.txt` disallows (`Disallow: /*?*`);
* it answers 403 without a browser user agent.

So it cannot supply history, but it can supply the future. Capture starts
recording positions now, so a later phase can test crew-chief effects once
enough seasons have accumulated.

## Call-level play-by-play — feasibility only

The brief notes the NBA has attached responsible referee names to foul and
violation calls since the 2015 playoffs. Feasibility here:

* **Basketball-Reference play-by-play does not carry it.** A sampled PBP page
  has 34 foul rows, typed (`Shooting foul by A. Davis (drawn by N. Jokić)`),
  and **zero** referee links inside those rows. Foul *type* is available;
  attribution to an official is not.
* **The NBA's own play-by-play does carry it, on the host that blocked us.**

Verdict: **not feasible today**, and blocked by access rather than by data.
The richer layer stays out of scope until simple crew tendencies first show
incremental value.

## Join

Deterministic, never fuzzy. Monthly schedule pages give each game's box score
address plus its date and both team codes; the key is the Eastern date plus
both canonical team codes. Basketball-Reference codes that differ (`PHO`,
`BRK`, `CHO`, and historical `NJN`, `NOH`, `SEA`, `VAN`, `CHH`) are mapped
explicitly; an unrecognised code is reported, not guessed. A key matching more
than one source row is recorded as ambiguous rather than resolved arbitrarily.

## Prospective capture

`capture_referee_assignments` reads the page, archives it, and appends to a
per-day ledger. Assignments change during the day, so:

* both states are kept, each with its own `first_observed_at_utc`;
* `known_at(matchup, cutoff)` returns only what was observed by that cutoff, so
  a crew discovered at 14:00 can never be backfilled into a T-6h prediction;
* re-observing an unchanged crew records nothing and preserves the original
  first-observed time, which is what makes restarts harmless.

It runs inside the collector every 30 minutes and is **strictly additive**: the
whole call is wrapped, failures are counted, and nothing it does can interrupt
report or market capture. `--no-referees` switches it off entirely.

```bash
python -m nba_prediction_market.pipelines.capture_referee_assignments
```

An empty table is a normal result — offseason, and every morning before the
league posts.
