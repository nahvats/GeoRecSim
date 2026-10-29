# Main experiment results: CSV schema

## Zip files

| zip | saved | agents | days | seeds | conditions |
|---|---|---|---|---|---|
| `main_study_recent_2026-10-28.zip` | 2026-10-26 to 2026-10-28 | 2,000 | 100 | 42–46 | 6 |
| `main_study_previous_2026-10-14.zip` | 2026-10-13 to 2026-10-14 | 2,500 | 60 | 42–46 | 5 (no PUP+RM) |

Conditions: `No RS`, `Oracle`, `Personalized`, `Personalized + PUP`, `Personalized + RM`, `Personalized + PUP+RM` (recent only).

Each zip contains one folder per city (`chicago, dc, houston, la, miami, nyc, seattle, sf_bay`), and each folder contains three files:

| file | one row per | rows per city (recent) | rows per city (previous) |
|---|---|---|---|
| `results.csv` | condition | 6 | 5 |
| `daily.csv` | condition × seed × day | 6 × 5 × 100 = 3,000 | 5 × 5 × 60 = 1,500 |
| `agents.csv` | condition × seed × agent | 6 × 5 × 2,000 = 60,000 | 5 × 5 × 2,500 = 62,500 |

Every file starts with a `city` column, so files from different cities can be stacked.

### Seeds

- Every condition in a city runs on the same 5 seeds. At a given seed, all conditions have the same agents and the same random draws, so differences between conditions come from the recommender alone.
- The same seed number in two cities gives two different runs, because each city has its own population.

### Last day vs. all days

- **Last day** = the final simulated day (day 100 recent, day 60 previous). Table 1 and Table 2 are computed from it.
- **All days** = every day of the run.

### Utility

Utility is the net utility of one leisure trip. Higher is better, and below 0 means the agent was worse off for going.

---

## results.csv

This is the website's Table 1 and Table 2 in one row per condition. Each value is the average over the 5 seeds, and all values are from the last day.

| column | meaning |
|---|---|
| `city` | metro |
| `condition` | recommender condition |
| `mean_utility` | mean leisure-trip utility (Ū) |
| `sigma_u` | standard deviation of the 5 per-seed `mean_utility` values (σ_U) |
| `neg_rate` | share of leisure trips with utility < 0 |
| `abstention_rate` | share of recommendation opportunities where the recommender showed nothing (1.0 for No RS) |
| `gini` | Gini of trip utilities |
| `n_leisure_trips` | leisure trips, summed over the 5 seeds |
| `harmed_pct` | Table 2: share of matched agents worse off than under No RS |
| `improved_pct` | Table 2: share of matched agents better off than under No RS |
| `mean_orc` | Table 2: mean utility lost (No RS minus this condition) over the harmed agents |
| `n_matched` | Table 2: number of matched agents, averaged over seeds |

- The Table 2 columns are empty for `No RS`, because it is the baseline.
- `mean_orc` is empty if no agent was harmed.
- **Matched agents:**
  - **Recent zip:** agents who took a recommendation on the last day and also made a leisure trip on the last day of the No RS run with the same seed.
  - **Previous zip:** every agent with a last-day leisure trip in both runs, whether or not they took a recommendation. The previous set predates the acceptance record, so Table 2 is **not** directly comparable between the two zips.

## daily.csv

One row per day of each run.

| column | meaning |
|---|---|
| `city`, `condition`, `seed`, `day` | identify the row; `day` starts at 1 |

**That day only:**

| column | meaning |
|---|---|
| `leisure_trips` | leisure trips made that day |
| `acceptance_rate` | share of leisure trips that went to a recommended venue (0 for No RS) |
| `mean_utility` | mean leisure-trip utility |
| `taste_match` | share of leisure trips to a venue matching the agent's hidden taste |
| `taste_match_recommended` | `taste_match` for trips to a recommended venue |
| `taste_match_organic` | `taste_match` for trips the agent chose itself |
| `recommended_trips` | number of trips to a recommended venue |
| `organic_trips` | number of trips the agent chose itself |

**Running totals from day 1 through this day.** Visits to venues are accumulated, so these values grow over the run:

| column | meaning |
|---|---|
| `footfall_gini` | Gini of visit counts across the venues visited so far (not the same as `gini` in results.csv) |
| `effective_places` | 1 / Σ(share²): how many equally busy venues would give the same spread |
| `top10_share` | share of all visits that went to the 10 busiest venues |
| `distinct_visited` | number of venues visited at least once |
| `coverage` | `distinct_visited` / `n_places` |
| `repeat_rate` | share of visits that were not a venue's first visit |
| `total_visits` | total leisure visits so far |
| `n_places` | number of venues in the city's catalog (constant) |
| `visits_per_place` | `total_visits` / `n_places` |

## agents.csv

One row per agent in each run.

| column | meaning |
|---|---|
| `city`, `condition`, `seed`, `agent_id` | identify the row; `agent_id` is the same agent across conditions at a given seed |
| `income` | income band |
| `trust` | trust-in-recommendations quartile, Q1 (lowest) to Q4 |
| `paradigm` | decision rule for mode choice: `utility`, `satisficing`, `regret`, `prospect`, `habit` |
| `car_access` | `own car` or `none` |
| `home_lat`, `home_lon` | home location |
| `last_day_utility` | utility of the agent's leisure trip on the last day; empty if none |
| `avg_utility_all_days` | mean leisure-trip utility over all days; empty if the agent never made a leisure trip |
| `took_recommendation_last_day` | 1 = the last-day leisure trip went to a recommended venue, 0 = it did not; empty if no last-day leisure trip. **Recent zip only**; always empty in the previous zip |

- Columns `income` through `home_lon` are identical across conditions at the same seed.
- `income` mixes two label sets (`<35k`, `35-75k`, `125k+` and the Census bands like `$50,000 - $74,999`), because agents come from two data sources. Values are exported as stored.
- Agents make at most one leisure trip on the last day, so `last_day_utility` is that trip's utility.
