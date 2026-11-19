# The recommender layer

## One scorer, many platforms

A study always includes two controls, **No-RS** and **Oracle**, plus up to
five recommenders you build in the sidebar. Every recommender is the same
scorer with different knob values. The presets ("Proximity-led",
"Footfall-led", "Viral", …) differ only in which weights they turn up or zero
out, not in their machinery. They are defined in
`welfare_rs/recommender_systems.py` (`RECOMMENDER_PRESETS`).

| Knob | What it does |
|---|---|
| `w_rating` / `w_reviews` | how prominent a venue is; both change as agents leave feedback |
| `w_relevance` | how well the venue's category fits the request; the same for everyone |
| `w_proximity` | `exp(-d / distance_scale_km)` on straight-line distance |
| `w_popularity` | live footfall relative to the busiest candidate; this is the feedback loop |
| `w_personalization` | learned per-agent affinity; the only personal signal |
| `popularity_gamma` | exponent on relative footfall: below 1 flattens, above 1 concentrates |
| `learning_rate` | how quickly feedback moves the user model |
| `welfare_gate` | optional PUP or RM filter applied on top of the ranking |

Weights are renormalized to sum to 1 and every component is bounded in
[0, 1], so scores are comparable across configurations. This matters because
acceptance reads `score - 0.5`. An earlier design had one recommender min-max
normalizing (top candidate always near 1.0) while another emitted a raw
weighted sum topping out near 0.75, so the first won every cross-platform
comparison for reasons that were pure scale.

On a cold start the popularity column doesn't exist yet. Its weight is spread
over the other components instead of being counted as zero, which would
otherwise cap a popularity-heavy recommender at a low score for the whole
warm-up.

Note that the recommender ranks on straight-line distance while trips are
charged on the street network, so a suggestion can be cheap to score and
expensive to take.

## The Oracle control

Oracle is always run and is the one arm that isn't this scorer. It is the
perfect-knowledge upper bound from Uğurel & Yabe (2026, condition 5): it reads
the agent's true preferences, hidden tastes included, and offers the venue of
the chosen subtype with the highest net utility `U = V − C`. `V` is the
activity benefit the trip is credited with on arrival, and `C` is the agent's
own expected round-trip cost, the same one it ranks its own options with.

Like the paper's oracle, it ranks on the ex-ante view and does not see
congestion or the agent's own pick, so it can still harm some agents. Its score
is a neutral 0.5, so the quality term in acceptance is zero and acceptance
works exactly as for every other arm. The gap between Oracle and the other arms
is therefore ranking knowledge alone. Weights and the welfare gate are ignored.
See `welfare_rs/oracle_rs.py`.

## Welfare gates

A recommender can optionally filter its ranking through a welfare gate before
showing it (`welfare_rs/welfare_layer.py`):

- **PUP (positive utility probability)** keeps venues whose chance of leaving
  the agent better off clears a threshold. Its uncertainty is learned from the
  platform's own prediction errors.
- **RM (regret minimization)** drops venues the agent is likely to regret
  relative to the alternatives.

## Hidden tastes

Each venue carries taste tags recovered from its business name ("Joe's Pizza"
becomes `pizza`). Each agent holds a hidden favourite per family (`dining`,
`cafe`, `fitness`, `outdoor`, `culture`, `music`), drawn from the catalog's own
supply mix with a tempering exponent `beta` (`welfare_rs/tastes.py`).

Recommenders can't see the taste. It moves the agent's realised enjoyment of a
visit, that enjoyment drives like/dislike feedback, and personalization has to
infer the taste from the feedback. Taste discovery is a real learning problem,
not a lookup.

Two caveats:

- Only about half of real venues name their type, and which ones do is biased
  (pizza and Thai places label themselves; Korean and New American places
  almost never do). Unlabelled venues are given an imputed tag from the
  observed distribution for their family, keyed on a hash of the place id,
  rather than being treated as "definitely not Thai". The alternative would
  quietly penalize half the catalog.
- Tastes are sampled from how often each cuisine can be identified by name,
  which is a proxy for demand, not a measurement of it.
