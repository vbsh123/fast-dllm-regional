"""Small, model-independent scheduler for balanced regional decoding.

The model still evaluates the complete Dream canvas.  This module controls only
which fixed response regions may consume their local transfer quota after a
forward pass.  Progress is the number of actually revealed tokens; a scheduled
zero-token transition does not count as progress.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegionState:
    index: int
    start: int
    end: int
    schedule_step: int = 0
    deferrals: int = 0

    @property
    def size(self) -> int:
        return self.end - self.start


def build_regions(generation_length: int, region_size: int) -> list[RegionState]:
    if generation_length <= 0:
        raise ValueError("generation_length must be positive")
    if region_size <= 0:
        raise ValueError("region_size must be positive")
    return [
        RegionState(index=index, start=start, end=min(start + region_size, generation_length))
        for index, start in enumerate(range(0, generation_length, region_size))
    ]


def linear_transfer_count(
    remaining_masks: int,
    *,
    schedule_step: int,
    local_steps: int,
    eps: float,
) -> int:
    """Dream's linear transfer schedule, evaluated on one region."""
    if remaining_masks <= 0:
        return 0
    if local_steps <= 0:
        raise ValueError("local_steps must be positive")
    if not 0 <= schedule_step < local_steps:
        raise ValueError("schedule_step must index an unfinished local schedule")
    if schedule_step == local_steps - 1:
        return remaining_masks
    delta = (1.0 - eps) / local_steps
    current_time = 1.0 - schedule_step * delta
    next_time = 1.0 - (schedule_step + 1) * delta
    return int(remaining_masks * (1.0 - next_time / current_time))


def startup_force_reason(
    state: RegionState,
    *,
    remaining_masks: int,
    deferral_until_revealed: int,
    max_region_deferrals: int,
) -> str | None:
    """Return the region-local reason that bypasses startup deferral."""
    if deferral_until_revealed < 0:
        raise ValueError("deferral_until_revealed must be non-negative")
    if max_region_deferrals < 0:
        raise ValueError("max_region_deferrals must be non-negative")
    revealed = state.size - remaining_masks
    if revealed >= deferral_until_revealed:
        return "deferral_window_closed"
    if state.deferrals >= max_region_deferrals:
        return "region_deferral_limit"
    return None


def controlled_regions(
    states: list[RegionState],
    *,
    remaining_masks: list[int],
    local_steps: int,
    max_progress_gap: int,
    max_region_exclusive: int | None = None,
    progress_gap_exempt_children: set[int] | None = None,
) -> tuple[list[int], set[int], set[int]]:
    """Return active, blocked, and urgency-forced region indices.

    Adjacent regions are coupled by actual revealed-token progress.  A child
    cannot lead its parent, and a parent cannot lead its child by
    ``max_progress_gap`` or more.  The lagging endpoint becomes urgent so its
    startup confidence deferral may be bypassed to release backpressure.
    """
    if len(states) != len(remaining_masks):
        raise ValueError("remaining_masks must contain one value per region")
    if max_progress_gap < 0:
        raise ValueError("max_progress_gap must be non-negative")

    eligible = {
        state.index
        for state, remaining in zip(states, remaining_masks)
        if remaining > 0
        and state.schedule_step < local_steps
        and (max_region_exclusive is None or state.index < max_region_exclusive)
    }
    blocked: set[int] = set()
    urgent: set[int] = set()
    progress = [state.size - remaining for state, remaining in zip(states, remaining_masks)]

    for parent_index in range(len(states) - 1):
        child_index = parent_index + 1
        if parent_index not in eligible or child_index not in eligible:
            continue
        parent_progress = progress[parent_index]
        child_progress = progress[child_index]
        if child_progress > parent_progress:
            blocked.add(child_index)
            urgent.add(parent_index)
        elif (
            parent_progress - child_progress >= max(1, max_progress_gap)
            and child_index not in (progress_gap_exempt_children or set())
        ):
            blocked.add(parent_index)
            urgent.add(child_index)

    urgent.difference_update(blocked)
    active = sorted(eligible - blocked)
    return active, blocked, urgent
