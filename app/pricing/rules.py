"""Deterministic application of configurable business rules.

The rules layer turns an unconstrained, revenue-optimal price into a publishable
price by enforcing hard safety bounds (absolute min/max, minimum margin),
smoothing constraints (maximum day-over-day change), occupancy-driven pressure,
and optional manual overrides. Every binding constraint is recorded so the
recommendation is fully auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.pricing import AppliedConstraint, BusinessRules
from app.utils.exceptions import PricingError


@dataclass
class RuleResult:
    """Outcome of applying :class:`BusinessRules` to a price."""

    price: float
    effective_floor: float
    effective_ceiling: float
    manual_override_applied: bool
    applied_constraints: list[AppliedConstraint] = field(default_factory=list)


def round_to_increment(value: float, increment: float) -> float:
    """Round ``value`` to the nearest multiple of ``increment``."""
    if increment <= 0:
        return round(value, 2)
    return round(round(value / increment) * increment, 2)


def _effective_bounds(
    rules: BusinessRules, previous_price: float | None, occupancy: float
) -> tuple[float, float, list[AppliedConstraint]]:
    """Compute the floor/ceiling window and the constraints that produced it."""
    constraints: list[AppliedConstraint] = []

    hard_floor = max(rules.min_rate, rules.margin_floor())
    hard_ceiling = rules.max_rate
    if hard_floor > hard_ceiling:
        raise PricingError(
            "Infeasible rules: minimum (margin/min_rate) exceeds max_rate.",
            details={"hard_floor": hard_floor, "max_rate": hard_ceiling},
        )

    floor, ceiling = hard_floor, hard_ceiling

    if previous_price is not None and previous_price > 0:
        smooth_floor = previous_price * (1.0 - rules.max_daily_change_pct)
        smooth_ceiling = previous_price * (1.0 + rules.max_daily_change_pct)
        if smooth_floor > floor:
            floor = smooth_floor
            constraints.append(
                AppliedConstraint(
                    rule="max_daily_change_pct",
                    description=(
                        f"Down-move limited to {rules.max_daily_change_pct:.0%} of "
                        f"previous price ({previous_price:.2f})."
                    ),
                )
            )
        capped_ceiling = min(hard_ceiling, max(smooth_ceiling, floor))
        if capped_ceiling < ceiling:
            ceiling = capped_ceiling
            constraints.append(
                AppliedConstraint(
                    rule="max_daily_change_pct",
                    description=(
                        f"Up-move limited to {rules.max_daily_change_pct:.0%} of "
                        f"previous price ({previous_price:.2f})."
                    ),
                )
            )

        # Occupancy-driven directional pressure.
        if occupancy >= rules.occupancy_ceiling and previous_price > floor:
            floor = min(previous_price, ceiling)
            constraints.append(
                AppliedConstraint(
                    rule="occupancy_ceiling",
                    description=(
                        f"Occupancy {occupancy:.0%} >= {rules.occupancy_ceiling:.0%}; "
                        "price held at or above previous to protect remaining inventory."
                    ),
                )
            )
        elif occupancy <= rules.occupancy_floor and previous_price < ceiling:
            ceiling = max(previous_price, floor)
            constraints.append(
                AppliedConstraint(
                    rule="occupancy_floor",
                    description=(
                        f"Occupancy {occupancy:.0%} <= {rules.occupancy_floor:.0%}; "
                        "price capped at or below previous to stimulate demand."
                    ),
                )
            )

    return floor, ceiling, constraints


def apply_business_rules(
    optimal_price: float,
    rules: BusinessRules,
    *,
    previous_price: float | None = None,
    occupancy: float = 0.0,
) -> RuleResult:
    """Apply all business rules to ``optimal_price`` and return the final price."""
    if rules.manual_override is not None:
        clamped = min(max(rules.manual_override, rules.min_rate), rules.max_rate)
        final = round_to_increment(clamped, rules.rounding_increment)
        return RuleResult(
            price=final,
            effective_floor=rules.min_rate,
            effective_ceiling=rules.max_rate,
            manual_override_applied=True,
            applied_constraints=[
                AppliedConstraint(
                    rule="manual_override",
                    description=f"Manual override of {rules.manual_override:.2f} applied.",
                )
            ],
        )

    floor, ceiling, constraints = _effective_bounds(rules, previous_price, occupancy)

    final = optimal_price
    if final < floor:
        final = floor
        constraints.append(
            AppliedConstraint(
                rule="price_floor",
                description=f"Raised to effective floor {floor:.2f}.",
            )
        )
    elif final > ceiling:
        final = ceiling
        constraints.append(
            AppliedConstraint(
                rule="price_ceiling",
                description=f"Lowered to effective ceiling {ceiling:.2f}.",
            )
        )

    final = round_to_increment(final, rules.rounding_increment)
    final = min(max(final, rules.min_rate), rules.max_rate)
    return RuleResult(
        price=final,
        effective_floor=round(floor, 2),
        effective_ceiling=round(ceiling, 2),
        manual_override_applied=False,
        applied_constraints=constraints,
    )
