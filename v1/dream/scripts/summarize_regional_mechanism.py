#!/usr/bin/env python3
"""Summarize regional scheduler telemetry embedded in evaluation logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PREFIX = "generation_stats: "


def load_stats(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(PREFIX):
                    row = json.loads(line[len(PREFIX) :])
                    if "startup_mechanism" in row:
                        rows.append(row)
    return rows


def safe_ratio(numerator: int | float, denominator: int | float):
    return numerator / denominator if denominator else None


def weighted_probability_mean(rows: list[dict], section: str) -> float | None:
    weighted_sum = 0.0
    count = 0
    for row in rows:
        summary = row["commit_confidence"][section]
        item_count = int(summary["count"])
        if item_count and summary["mean"] is not None:
            weighted_sum += float(summary["mean"]) * item_count
            count += item_count
    return weighted_sum / count if count else None


def summarize(rows: list[dict]) -> dict:
    startup = [row["startup_mechanism"] for row in rows]
    concurrency = [row["concurrency_mechanism"] for row in rows]

    def total(items: list[dict], key: str) -> int:
        return sum(int(item.get(key, 0)) for item in items)

    candidate_events = total(startup, "candidate_update_events")
    deferred_events = total(startup, "deferred_update_events")
    committed_events = total(startup, "committed_update_events")
    gap_forced = total(startup, "gap_forced_low_confidence_commit_events")
    global_forced = total(
        startup, "global_deadlock_forced_low_confidence_commit_events"
    )
    total_nfe = sum(int(row["nfe"]) for row in rows)
    committed_probability_count = sum(
        int(
            row["commit_confidence"][
                "all_committed_token_top1_probability"
            ]["count"]
        )
        for row in rows
    )
    first_commit_nfes = [
        region["first_commit_nfe"]
        for item in startup
        for region in item["per_region"]
        if region["first_commit_nfe"] is not None
    ]
    startup_complete_nfes = [
        region["startup_complete_nfe"]
        for item in startup
        for region in item["per_region"]
        if region["startup_complete_nfe"] is not None
    ]

    per_region = defaultdict(
        lambda: {
            "examples": 0,
            "candidate_update_events": 0,
            "deferred_update_events": 0,
            "confidence_pass_commit_events": 0,
            "gap_forced_commit_events": 0,
            "global_deadlock_forced_commit_events": 0,
            "bootstrap_tokens_committed": 0,
            "first_commit_nfes": [],
            "startup_complete_nfes": [],
        }
    )
    for item in startup:
        for region in item["per_region"]:
            aggregate = per_region[int(region["region_index"])]
            aggregate["examples"] += 1
            for key in (
                "candidate_update_events",
                "deferred_update_events",
                "confidence_pass_commit_events",
                "gap_forced_commit_events",
                "global_deadlock_forced_commit_events",
                "bootstrap_tokens_committed",
            ):
                aggregate[key] += int(region[key])
            if region["first_commit_nfe"] is not None:
                aggregate["first_commit_nfes"].append(region["first_commit_nfe"])
            if region["startup_complete_nfe"] is not None:
                aggregate["startup_complete_nfes"].append(
                    region["startup_complete_nfe"]
                )

    per_region_rows = []
    for region_index, item in sorted(per_region.items()):
        examples = item["examples"]
        per_region_rows.append(
            {
                "region_index": region_index,
                "examples": examples,
                "mean_candidate_updates": item["candidate_update_events"] / examples,
                "mean_deferrals": item["deferred_update_events"] / examples,
                "mean_confidence_pass_commits": (
                    item["confidence_pass_commit_events"] / examples
                ),
                "mean_gap_forced_commits": (
                    item["gap_forced_commit_events"] / examples
                ),
                "mean_global_forced_commits": (
                    item["global_deadlock_forced_commit_events"] / examples
                ),
                "mean_first_commit_nfe": (
                    sum(item["first_commit_nfes"])
                    / len(item["first_commit_nfes"])
                    if item["first_commit_nfes"]
                    else None
                ),
                "mean_startup_complete_nfe": (
                    sum(item["startup_complete_nfes"])
                    / len(item["startup_complete_nfes"])
                    if item["startup_complete_nfes"]
                    else None
                ),
            }
        )

    configurations = sorted(
        {
            (
                row["region_size"],
                row["local_steps"],
                row["max_progress_gap"],
                row["deferral_threshold"],
                row["deferral_until_revealed"],
                row["stop_mode"],
            )
            for row in rows
        },
        key=str,
    )
    return {
        "examples": len(rows),
        "configurations": [
            {
                "region_size": item[0],
                "local_steps": item[1],
                "max_progress_gap": item[2],
                "deferral_threshold": item[3],
                "deferral_until_revealed": item[4],
                "stop_mode": item[5],
            }
            for item in configurations
        ],
        "mean_nfe": total_nfe / len(rows),
        "startup": {
            "candidate_update_events": candidate_events,
            "deferred_update_events": deferred_events,
            "confidence_pass_commit_events": total(
                startup, "confidence_pass_commit_events"
            ),
            "gap_forced_low_confidence_commit_events": gap_forced,
            "global_forced_low_confidence_commit_events": global_forced,
            "bootstrap_tokens_committed": total(
                startup, "bootstrap_tokens_committed"
            ),
            "deferral_rate": safe_ratio(deferred_events, candidate_events),
            "forced_rate_per_committed_update": safe_ratio(
                gap_forced + global_forced, committed_events
            ),
            "mean_first_commit_nfe": (
                sum(first_commit_nfes) / len(first_commit_nfes)
                if first_commit_nfes
                else None
            ),
            "mean_startup_complete_nfe": (
                sum(startup_complete_nfes) / len(startup_complete_nfes)
                if startup_complete_nfes
                else None
            ),
        },
        "concurrency": {
            "mean_committing_regions_per_forward": safe_ratio(
                total(concurrency, "committing_region_events"), total_nfe
            ),
            "fraction_forwards_with_multiple_committing_regions": safe_ratio(
                total(concurrency, "forwards_with_multiple_committing_regions"),
                total_nfe,
            ),
            "fraction_forwards_with_no_scheduler_commit": safe_ratio(
                total(concurrency, "forwards_with_no_scheduler_commit"),
                total_nfe,
            ),
        },
        "confidence": {
            "mean_all_committed_token_top1_probability": (
                weighted_probability_mean(
                    rows, "all_committed_token_top1_probability"
                )
            ),
            "mean_startup_committed_token_top1_probability": (
                weighted_probability_mean(
                    rows, "startup_committed_token_top1_probability"
                )
            ),
            "fraction_committed_below_deferral_threshold": safe_ratio(
                sum(
                    int(
                        row["commit_confidence"][
                            "committed_tokens_below_deferral_threshold"
                        ]
                    )
                    for row in rows
                ),
                committed_probability_count,
            ),
            "fraction_committed_at_least_fast_reference_0_9": safe_ratio(
                sum(
                    int(
                        row["commit_confidence"][
                            "committed_tokens_at_least_fast_reference_0_9"
                        ]
                    )
                    for row in rows
                ),
                committed_probability_count,
            ),
        },
        "maximum_absolute_adjacent_progress_gap": max(
            (
                row["progress_balance"]["maximum_absolute_adjacent_gap"]
                for row in rows
                if row["progress_balance"]["maximum_absolute_adjacent_gap"]
                is not None
            ),
            default=None,
        ),
        "per_region": per_region_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    rows = load_stats(args.logs)
    if not rows:
        raise SystemExit("no regional generation_stats records found")
    print(json.dumps(summarize(rows), indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
