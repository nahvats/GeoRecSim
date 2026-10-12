"""Oracle recommender — the perfect-knowledge upper bound.

Uğurel & Yabe (2026), *Towards welfare-oriented recommendations in
activity-travel behavior*, §4 condition 5: "A hypothetical recommender with
perfect knowledge of each agent's true preference vector β_i, including their
value of time, time and cost sensitivities, leisure preferences, and
subtype-specific utility bonuses. It computes the exact net utility
U_ik = V_ik − C_ik for every candidate and returns the optimal ranking. This
condition provides an upper bound on achievable welfare."

Mapped onto this model, for the subtype the agent has already chosen:

``V_ik``  :meth:`Simulation.realised_activity_utility` — the benefit the trip
          is credited with on arrival, latent taste match included. The other
          recommenders cannot observe the taste term; the oracle reads it.
``C_ik``  :meth:`Agent._price_trip` — the agent's own expected cost of
          getting there: time x its value of time + money, each mode weighted
          by the probability this agent takes it under its own decision rule,
          scaled by its trip-burden sensitivity. The same cost model the agent
          ranks its own options with and prices eta's dC with.

Like the paper's oracle it ranks on the ex-ante view: free-flow roads and the
expected mode mix, not the congestion the trip will meet or the mode actually
drawn at departure. It also does not see the agent's organic alternative. So it
can still leave an agent worse off than No RS would have, as the paper's oracle
did (27.8% harmed, its Table 2).

What it deliberately does NOT change is the acceptance machinery. Every
recommendation carries a neutral score of 0.5, so eta's quality term
``quality_weight * (score - 0.5)`` is zero, and acceptance still responds to
trust, autonomy, dC and memory exactly as under any other arm. The gap between
this arm and the rest is therefore ranking knowledge alone. Net utility is
reported in ``components`` for inspection.

It has nothing to learn, so private feedback is ignored. Public feedback (the
like/dislike that moves a venue's live rating) still flows, because the
simulation records it on the shared PlaceDynamics, not on the recommender.
"""

from __future__ import annotations

from typing import Dict, List

from .recommender_systems import Recommendation, RecommenderSystem, UserContext
from .utils import haversine_km

#: The score every oracle recommendation carries. Neutral on eta's quality term
#: (see the module docstring), and inside the [0, 1] range every arm's scores
#: share.
NEUTRAL_SCORE = 0.5


class OracleRecommender(RecommenderSystem):
    """Ranks every candidate of the subtype by the agent's true net utility.

    ``env`` is the owning Simulation, from which the oracle reads the agent
    itself — its tastes, value of time and cost sensitivities. That access is
    what no other recommender has.
    """

    def __init__(self, env, label: str = "Oracle"):
        super().__init__(name=label, catalog=env.place_catalog)
        self.env = env
        self._agent_by_id = {a.id: a for a in env.agents}

    def true_net_utility(self, agent, user: UserContext, leisure_subtype: str, place):
        """``(U, V, C)`` for sending ``agent`` to ``place``, or None when no mode
        it may use makes the trip."""
        cost = agent._price_trip(self.env, user.location, place.location, leisure_subtype)
        if cost is None:
            return None
        value = self.env.realised_activity_utility(agent, leisure_subtype, place.place_id)
        return value - cost, value, cost

    def recommend(
        self, user: UserContext, leisure_subtype: str, top_k: int = 5
    ) -> List[Recommendation]:
        agent = self._agent_by_id.get(user.user_id)
        if agent is None:
            # Every simulated agent is registered at construction, so this is a
            # wiring error, not a cold start. An oracle that quietly fell back
            # to some heuristic would report that heuristic as the upper bound.
            raise KeyError(f"Oracle has no agent {user.user_id}")

        candidates = self._filter_by_subtype(leisure_subtype)
        # Same straight-line reach bound the agent's organic draw uses
        # (Simulation.sample_poi). It only removes venues that pricing below
        # would reject anyway, since network distance is never shorter.
        reach_km = self.env.reach_limit_km(agent)
        if reach_km is not None:
            candidates = [p for p in candidates
                          if haversine_km(user.location, p.location) <= reach_km]

        scored = []
        for place in candidates:
            priced = self.true_net_utility(agent, user, leisure_subtype, place)
            if priced is None:
                continue
            net, value, cost = priced
            scored.append((net, place.place_id, place, value, cost))
        # Tie-break on place_id, as ConfigurableRecommender does, so equal
        # utilities are not decided by catalog order.
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [
            Recommendation(
                source=self.name,
                place=place,
                score=NEUTRAL_SCORE,
                components={"net_utility": net, "v_act": value, "c_travel": cost},
            )
            for net, _pid, place, value, cost in scored[: max(1, top_k)]
        ]

    def record_feedback(self, user_id, place_id, liked, feedback_strength=1.0,
                        learning_rate=None) -> None:
        """Nothing to learn: the oracle already knows the agent."""
        del user_id, place_id, liked, feedback_strength, learning_rate


class OracleOrchestrator:
    """Adapts :class:`OracleRecommender` to the orchestrator interface.

    Returns a one-element list. Agent.plan_day takes ``max`` by score over
    what comes back, and every oracle score is the same neutral value, so the
    optimum is the only element offered — the choice cannot fall to tie order.
    """

    def __init__(self, oracle: OracleRecommender):
        self.oracle = oracle

    @property
    def name(self) -> str:
        return self.oracle.name

    def recommend(
        self, user: UserContext, leisure_subtype: str, top_k_per_system: int = 5
    ) -> Dict[str, List[Recommendation]]:
        del top_k_per_system  # only the optimum is ever acted on; see above
        return {self.oracle.name: self.oracle.recommend(user, leisure_subtype, top_k=1)}

    def record_feedback(self, user_id, source, place_id, liked,
                        feedback_strength: float = 1.0) -> None:
        del source
        self.oracle.record_feedback(user_id, place_id, liked, feedback_strength)


def oracle_factory(label: str = "Oracle"):
    """A ``recommender_factory`` for :class:`Simulation` that swaps the built
    recommender for the oracle. It needs the simulation's agents, which exist
    by the time the factory runs."""

    def factory(sim, _base_stack):
        return OracleOrchestrator(OracleRecommender(sim, label=label))

    return factory
