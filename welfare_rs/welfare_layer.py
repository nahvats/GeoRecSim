"""Utility-based welfare layer for the recommender system stack.

Implements:
- Continuous feedback signal (extends binary like/dislike → satisfaction + travel cost)
- Travel cost estimation from feedback history
- PUP (Positive Utility Probability) gating
- RM (Regret Minimization) gating
- WelfareAwareOrchestrator: wraps LeisureRSOrchestrator with welfare constraints
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import params
from .experiment_harness import leisure_net_utility
from .recommender_systems import (
    LeisureRSOrchestrator,
    Recommendation,
    UserContext,
)
from .utils import haversine_km


# ── Continuous feedback ──────────────────────────────────────────────────────

@dataclass
class ContinuousFeedback:
    """Extended feedback signal beyond binary like/dislike.

    Attributes:
        user_id: Agent identifier.
        place_id: Place that was visited.
        source: Recommender source name.
        satisfaction: Continuous satisfaction score in [-1, 1].
        realized_travel_cost: Actual travel cost experienced (generalized cost).
        activity_utility: Realized activity utility.
        travel_utility: Realized travel utility (negative = costly).
        net_utility: satisfaction proxy = activity_utility + travel_utility.
        realized_utility: the trip's net utility U = V - C/xi, as the welfare
            metric scores it (leisure_net_utility); the record carries the
            realized activity utility and generalized cost it is made of.
    """

    user_id: int
    place_id: str
    source: str
    satisfaction: float  # [-1, 1]
    realized_travel_cost: float  # generalized cost in $-equivalent
    activity_utility: float
    travel_utility: float
    net_utility: float
    realized_utility: float


def compute_continuous_feedback(
    trip,
    p_like: float,
    feedback_sensitivity: float = 1.0,
) -> ContinuousFeedback:
    """Convert a Trip + binary feedback signal into a continuous feedback record.

    The satisfaction score maps from the sigmoid probability p_like
    into a continuous [-1, 1] range, preserving the agent's nuanced
    evaluation rather than collapsing to binary.

    Parameters
    ----------
    trip : Trip
        The completed trip with utility components.
    p_like : float
        The probability of liking (from agent's feedback evaluation).
    feedback_sensitivity : float
        Agent-specific sensitivity multiplier.

    Returns
    -------
    ContinuousFeedback
        Continuous feedback record.
    """
    # Map p_like ∈ [0, 1] → satisfaction ∈ [-1, 1]
    satisfaction = (2.0 * p_like - 1.0) * min(2.0, max(0.2, feedback_sensitivity))
    satisfaction = max(-1.0, min(1.0, satisfaction))

    # Generalized travel cost: time cost + monetary cost
    # (travel_utility is already negative for costly trips)
    realized_travel_cost = trip.cost + (trip.travel_time_min / 60.0) * 15.0  # $15/hr default VOT

    net_utility = trip.activity_utility + trip.travel_utility

    return ContinuousFeedback(
        user_id=trip.agent_id,
        place_id=trip.recommended_place_id,
        source=trip.recommendation_source,
        satisfaction=satisfaction,
        realized_travel_cost=realized_travel_cost,
        activity_utility=trip.activity_utility,
        travel_utility=trip.travel_utility,
        net_utility=net_utility,
        realized_utility=leisure_net_utility(trip),
    )


# ── Travel cost estimation ───────────────────────────────────────────────────

class TravelCostEstimator:
    """The platform's estimate of a trip's travel cost, learned per user.

    Prices a candidate with the agent's own generalised-cost formula
    (Simulation._base_mode_utility), divided by ``gen_cost_denominator`` so it
    is in the same utility units as the activity benefit:

        Ĉ = ((τ̂ / 60) · v̂ + κ̂) / gen_cost_denominator

    What the platform cannot observe is estimated:

    - distance: straight-line km to the venue (the platform does not route);
    - mode: the mode this user took most often on the trips it has seen
      (``default_mode`` until it has seen one);
    - τ̂: that distance at the mode's speed, plus its wait;
    - κ̂: the mode's money cost over that distance (the metro's metered fare
      for a taxi, at the full fare);
    - v̂: the user's value of time, learned from feedback (below);
    - the time/cost weights and the trip-burden multiplier: 1, the population
      mean.

    Value of time is updated from continuous feedback: dissatisfaction with a
    long trip raises it 5%, satisfaction with one lowers it 3% (paper B.3).

    The uncertainty σ_i of PUP is learned from the platform's prediction errors
    (paper Eq. 17): see :meth:`get_user_uncertainty`.
    """

    def __init__(
        self,
        default_vot: float = 15.0,
        learning_rate: float = 0.15,
        metro: Optional[str] = None,
        default_mode: str = "car",
        sigma_prior: float = params.WELFARE_GATE_DEFAULTS["pup_sigma_prior"],
        sigma_prior_strength: float = params.WELFARE_GATE_DEFAULTS["pup_sigma_prior_strength"],
    ):
        self.default_vot = default_vot
        self.learning_rate = learning_rate
        self.metro = metro
        self.default_mode = default_mode
        self.sigma_prior = sigma_prior
        self.sigma_prior_strength = sigma_prior_strength
        # Per-user state: {user_id: {"vot": float, "avg_travel_cost": float,
        # "n": int, "modes": Counter, "sq_err": float, "n_err": int}}
        self.user_estimates: Dict[int, Dict[str, float]] = {}

    def _state(self, user_id: int) -> Dict[str, float]:
        return self.user_estimates.setdefault(
            user_id,
            {"vot": self.default_vot, "avg_travel_cost": 0.0, "n": 0, "modes": Counter(),
             "sq_err": 0.0, "n_err": 0},
        )

    def update(
        self,
        feedback: ContinuousFeedback,
        travel_time_min: float,
        monetary_cost: float,
        mode: Optional[str] = None,
    ) -> None:
        """Update user travel cost estimates from a feedback observation.

        ``mode`` is how the user travelled to the venue, which a platform that
        sent them there observes."""
        state = self._state(feedback.user_id)
        state["n"] += 1
        if mode:
            state["modes"][mode] += 1
        alpha = self.learning_rate

        # Update average travel cost with EMA
        realized_gc = monetary_cost + (travel_time_min / 60.0) * state["vot"]
        state["avg_travel_cost"] = (1 - alpha) * state["avg_travel_cost"] + alpha * realized_gc

        # Update VOT estimate: if user dislikes high-time trips, infer higher VOT
        # Simple heuristic: satisfaction < 0 on high-time trips → increase VOT
        if travel_time_min > 0 and feedback.satisfaction < -0.2:
            # User disliked a costly trip → they value time more
            state["vot"] = min(50.0, state["vot"] * (1 + 0.05 * abs(feedback.satisfaction)))
        elif travel_time_min > 20 and feedback.satisfaction > 0.3:
            # User liked a long trip → they don't mind travel as much
            state["vot"] = max(5.0, state["vot"] * (1 - 0.03 * feedback.satisfaction))

    def estimated_mode(self, user_id: int) -> str:
        """The mode this user took most often on the trips the platform saw
        (ties to the earliest seen), or ``default_mode`` before any."""
        state = self.user_estimates.get(user_id)
        modes = state.get("modes") if state else None
        if not modes:
            return self.default_mode
        return modes.most_common(1)[0][0]

    def _money(self, mode: str, distance_km: float, in_vehicle_min: float) -> float:
        """Money cost of ``mode`` over ``distance_km``, as the agent is charged."""
        if mode == "taxi":
            tariff = params.TAXI_FARES.get(self.metro) or params.TAXI_FARES["default"]
            miles = distance_km / 1.609344
            fare = tariff["flag"] + tariff["per_mile"] * miles
            first = tariff.get("first_mile_per_mile")
            if first is not None:
                fare += (first - tariff["per_mile"]) * min(miles, 1.0)
            return fare
        spec = params.MODE_PARAMS[mode]
        money = spec["fixed_cost"] + spec["cost_per_km"] * distance_km
        overage = spec.get("overage_per_min", 0.0)
        if overage:
            money += overage * max(0.0, in_vehicle_min - spec.get("included_min", 0.0))
        return money

    def estimate_travel_cost(self, user_id: int, distance_km: float) -> float:
        """Ĉ^travel_{ik} in utility units, for a venue ``distance_km`` away in a
        straight line (see the class docstring)."""
        return self.estimate_travel_costs(user_id, (distance_km,))[0]

    def estimate_travel_costs(self, user_id: int, distances_km: Sequence[float]) -> List[float]:
        """:meth:`estimate_travel_cost` for many venues at once: the user's mode
        and value of time are looked up once, not once per venue."""
        state = self.user_estimates.get(user_id)
        vot = self.default_vot if state is None else state["vot"]
        mode = self.estimated_mode(user_id)
        spec = params.MODE_PARAMS[mode]
        per_min = vot / 60.0
        speed = spec["speed_kmh"]
        wait = spec["wait_min"]
        denom = params.UTILITY_WEIGHTS["gen_cost_denominator"]
        out = []
        for d in distances_km:
            in_vehicle_min = d / speed * 60.0
            out.append(((in_vehicle_min + wait) * per_min + self._money(mode, d, in_vehicle_min)) / denom)
        return out

    def get_user_vot(self, user_id: int) -> float:
        """Return estimated VOT for a user."""
        state = self.user_estimates.get(user_id)
        return state["vot"] if state else self.default_vot

    def record_prediction_error(self, user_id: int, error: float) -> None:
        """Record one prediction error ``U - Û`` of this user's: the realized
        net utility of a recommendation they took minus the platform's estimate
        of it when it was offered."""
        state = self._state(user_id)
        state["sq_err"] += error * error
        state["n_err"] += 1

    def get_user_uncertainty(self, user_id: int) -> float:
        """σ_i of PUP (paper Eq. 17): the residual uncertainty of ``Û``,
        "learned from prediction errors in the feedback history".

            σ_i² = (κ · σ_0² + Σ_j e_j²) / (κ + n_i)

        the user's mean squared error ``e = U - Û`` over the n_i
        recommendations they took, shrunk toward the prior σ_0 with the weight
        of κ pseudo-observations. A user with no feedback gets σ_0, which is
        calibrated as the root mean squared error of Û over the Standard RS's
        accepted recommendations, the way RM's ε is calibrated from Standard's
        regrets. So σ_i is in the units of U and never rests on an arbitrary
        cold-start value: the previous 1/√n with a cold start of 1 was four
        times the measured error, and with no feedback it never moved, so a
        strict α abstained forever.
        """
        state = self.user_estimates.get(user_id)
        sq_err, n_err = (0.0, 0) if state is None else (state["sq_err"], state["n_err"])
        k = self.sigma_prior_strength
        return math.sqrt((k * self.sigma_prior ** 2 + sq_err) / (k + n_err))


# ── PUP and RM computation ───────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_pup(
    v_hat_act: float,
    c_hat_travel: float,
    sigma_i: float,
) -> float:
    """Compute Positive Utility Probability.

    PUP = Pr[U_{ik} ≥ 0] ≈ Φ((V̂^act - Ĉ^travel) / σ_i)

    Parameters
    ----------
    v_hat_act : float
        Estimated activity benefit.
    c_hat_travel : float
        Estimated travel cost.
    sigma_i : float
        User-specific uncertainty parameter.

    Returns
    -------
    float
        Probability in [0, 1] that net utility is non-negative.
    """
    if sigma_i <= 0:
        return 1.0 if v_hat_act >= c_hat_travel else 0.0
    z = (v_hat_act - c_hat_travel) / sigma_i
    return _normal_cdf(z)


def compute_regret(u_hat: float, candidate_u_hats: Sequence[float]) -> float:
    """Estimated regret of recommending a candidate (Uğurel & Yabe 2026, Eq. 18).

    R̂_{ik} = max_j Û_{ij} − Û_{ik}, with Û = V̂^act − Ĉ^travel, over the
    alternatives in ``candidate_u_hats``: the candidates the RS may offer and,
    optionally, the agent's organic alternative.

    Parameters
    ----------
    u_hat : float
        Estimated net utility of the candidate.
    candidate_u_hats : Sequence[float]
        Estimated net utilities of every alternative, this candidate included.

    Returns
    -------
    float
        Regret, non-negative.
    """
    return max(candidate_u_hats) - u_hat


def estimate_activity_utility(rec: Recommendation, leisure_subtype: str) -> float:
    """V̂^act: the agent's activity benefit, as the platform can estimate it.

    The agent is credited ``subtype base + motivation terms + taste bonus``
    (Simulation.realised_activity_utility). The platform uses the same formula:

    - the subtype's base benefit is a public parameter;
    - the motivation terms are private to the agent and identical for every
      venue of a subtype, so they are left out — they cancel in the regret,
      and leave PUP's level slightly conservative;
    - the taste bonus is ``match_bonus`` times the recommender's estimate of
      how much the agent will like the venue, in [0, 1] (0.5 when it has
      learned nothing): ``taste_estimate``, its personalization belief without
      the exploration bonus, or the personalization component for a
      recommender that does not report one.

    The ranking score is deliberately not used: rating and popularity are not
    part of the agent's utility, and proximity already enters through Ĉ.
    """
    base = params.LEISURE_SEGMENTS.get(leisure_subtype, {}).get("activity_utility", 0.0)
    fit = rec.components.get("taste_estimate", rec.components.get("personalization", 0.5))
    return base + params.TASTE_PARAMS["match_bonus"] * fit


# ── Welfare-aware orchestrator ───────────────────────────────────────────────

@dataclass
class WelfareAnnotatedRecommendation:
    """A recommendation annotated with welfare metrics."""

    recommendation: Recommendation
    v_hat_act: float
    c_hat_travel: float
    pup: float
    regret: float
    passed_pup: bool
    passed_rm: bool
    #: The best Û over every venue of the subtype: RM's benchmark.
    u_hat_best: Optional[float] = None


class WelfareAwareOrchestrator:
    """Wraps LeisureRSOrchestrator with PUP/RM gating.

    The pool is the ``candidate_pool`` best-scored venues of the subtype (all
    of them when None): the shortlist the platform is willing to offer. Each is
    annotated with Û = V̂ − Ĉ, its PUP and its regret (Uğurel & Yabe 2026,
    Eqs. 6 and 18)

        R̂_k = max_j Û_j − Û_k,   j over EVERY venue of the subtype,

    how far the candidate falls short of the best venue the platform knows of
    (the paper's "best available activity ... over the full set A"). The
    benchmark uses only what a platform has, its catalog and its own estimates,
    never the user's unobserved organic choice. It is also not the shortlist's
    own best, which would always pass with zero regret and make RM unable to
    abstain. The gate drops the candidates that fail and returns the
    ``top_k_per_system`` best-scored survivors, so the agent is offered the
    highest-scoring admissible venue, or nothing (Eqs. 9 and 10).

    Treatment modes:
    - "standard": pass-through (no welfare filtering)
    - "pup": filter recommendations where PUP < α
    - "rm": filter recommendations whose regret exceeds ε. At ε = 0 it offers
      a venue only when the shortlist holds the catalog's best, and abstains
      otherwise; as ε grows it becomes Standard.
    - "pup_rm": apply both constraints
    """

    def __init__(
        self,
        base_orchestrator: LeisureRSOrchestrator,
        travel_cost_estimator: TravelCostEstimator,
        mode: str = "standard",
        pup_alpha: float = params.WELFARE_GATE_DEFAULTS["pup_alpha"],
        rm_epsilon: float = params.WELFARE_GATE_DEFAULTS["rm_epsilon"],
        coord_scale_km: float = 1.0,
        coord_distance_km=haversine_km,
        candidate_pool: Optional[int] = params.WELFARE_GATE_DEFAULTS["candidate_pool"],
    ):
        self.base = base_orchestrator
        self.tce = travel_cost_estimator
        self.mode = mode
        self.pup_alpha = pup_alpha
        self.rm_epsilon = rm_epsilon
        self.coord_scale_km = coord_scale_km
        # Straight-line (lat, lon) distance in km for the travel-cost estimate.
        self.coord_distance_km = coord_distance_km
        # How many of the best-scored candidates the gate may offer.
        # None = every candidate of the subtype.
        self.candidate_pool = candidate_pool
        # Feedback history for continuous learning
        self.feedback_history: List[ContinuousFeedback] = []
        # user_id -> {place_id: Û} of the venues last offered to the user, so
        # the feedback on the one taken can be scored against its estimate.
        self._offered_u_hat: Dict[int, Dict[str, float]] = {}

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k_per_system: int = 5,
    ) -> Dict[str, List[WelfareAnnotatedRecommendation]]:
        """Get welfare-annotated recommendations.

        Annotates the candidate pool from the base orchestrator with PUP and
        regret, filters it according to the active mode, and returns the
        ``top_k_per_system`` best-scored survivors per source.
        """
        pool = max(self.candidate_pool or sys.maxsize, top_k_per_system)
        # Every candidate, not just the pool: RM's benchmark ranges over all
        # of them.
        base_recs = self.base.recommend(user, leisure_subtype, sys.maxsize)
        sigma_i = self.tce.get_user_uncertainty(user.user_id)

        annotated: Dict[str, List[WelfareAnnotatedRecommendation]] = {}
        for source, recs in base_recs.items():
            dists = [
                self.coord_distance_km(user.location, rec.place.location) * self.coord_scale_km
                for rec in recs
            ]
            v_hats = [estimate_activity_utility(rec, leisure_subtype) for rec in recs]
            c_hats = self.tce.estimate_travel_costs(user.user_id, dists)
            u_hats = [v - c for v, c in zip(v_hats, c_hats)]

            # RM's benchmark: the best estimate over the whole catalog, taken
            # once here rather than inside compute_regret for every candidate.
            u_hat_best = max(u_hats) if u_hats else None
            best = (u_hat_best,)
            u_hats = u_hats[:pool]
            estimates = list(zip(recs, v_hats, c_hats))[:pool]
            annotated_list = []
            for (rec, v_hat_act, c_hat_travel), u_hat in zip(estimates, u_hats):
                pup = compute_pup(v_hat_act, c_hat_travel, sigma_i)
                regret = compute_regret(u_hat, best)

                passed_pup = pup >= self.pup_alpha
                passed_rm = regret <= self.rm_epsilon

                annotated_list.append(
                    WelfareAnnotatedRecommendation(
                        recommendation=rec,
                        v_hat_act=v_hat_act,
                        c_hat_travel=c_hat_travel,
                        pup=pup,
                        regret=regret,
                        passed_pup=passed_pup,
                        passed_rm=passed_rm,
                        u_hat_best=u_hat_best,
                    )
                )

            # Apply welfare filter
            if self.mode == "pup":
                annotated_list = [a for a in annotated_list if a.passed_pup]
            elif self.mode == "rm":
                annotated_list = [a for a in annotated_list if a.passed_rm]
            elif self.mode == "pup_rm":
                annotated_list = [a for a in annotated_list if a.passed_pup and a.passed_rm]
            # "standard" → no filtering

            # The pool arrives sorted by score, and filtering keeps that order.
            annotated[source] = annotated_list[: max(1, top_k_per_system)]

        self._offered_u_hat[user.user_id] = {
            a.recommendation.place.place_id: a.v_hat_act - a.c_hat_travel
            for recs in annotated.values() for a in recs
        }
        return annotated

    def record_continuous_feedback(
        self,
        feedback: ContinuousFeedback,
        travel_time_min: float,
        monetary_cost: float,
        mode: Optional[str] = None,
    ) -> None:
        """Record continuous feedback and update all learning components."""
        self.feedback_history.append(feedback)

        # Update travel cost estimator
        self.tce.update(feedback, travel_time_min, monetary_cost, mode=mode)
        # Score the estimate the venue was offered with against the outcome.
        u_hat = self._offered_u_hat.get(feedback.user_id, {}).get(feedback.place_id)
        if u_hat is not None:
            self.tce.record_prediction_error(feedback.user_id, feedback.realized_utility - u_hat)

    def record_feedback(self, user_id: int, source: str, place_id: str, liked: bool, feedback_strength: float = 1.0) -> None:
        """Backward-compatible binary feedback passthrough."""
        self.base.record_feedback(
            user_id=user_id,
            source=source,
            place_id=place_id,
            liked=liked,
            feedback_strength=feedback_strength,
        )
