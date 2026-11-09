<h1 align="center">GeoRecSim</h1>

<p align="center">
  <b>What happens to a city when an app decides where everyone goes?</b>
</p>

<p align="center">
  <a href="https://airecsim.cusp.nyu.edu/"><img src="https://img.shields.io/badge/Live%20demo-airecsim.cusp.nyu.edu-57068c?style=for-the-badge" alt="Live demo"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/backend-FastAPI-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/frontend-React%20%2B%20deck.gl-61dafb" alt="React + deck.gl">
  <img src="https://img.shields.io/badge/maps-OpenStreetMap-7ebc6f" alt="OpenStreetMap">
</p>

---

When you look for dinner, a coffee shop or a gym, an app usually picks the
shortlist. Those apps don't just reflect what people want, they shape it. The
places that get recommended get the visits, and the visits make them look even
more worth recommending. Add that up across a whole city and it changes which
neighborhoods get business, how far people travel, and who ends up better off.

**GeoRecSim** is a simulation for studying that effect. It builds a population
of people for a real US city, puts them on the city's actual street network,
and lets them decide where to go after work, either on their own or with a
recommender's help. You choose how the recommender ranks places, run the city
for days or weeks, and compare what happened against the same city with no
recommender at all.

**[Try it in your browser →](https://airecsim.cusp.nyu.edu/)** No install needed.

<p align="center">
  <img src="docs/img/georecsim-overview.png" alt="GeoRecSim overview: data sources feed a spatial environment of agents and venues; a tunable recommender ranks venues for each agent; the run reports footfall and visitor concentration, net utility by income, trip costs and activity benefit over time." width="100%">
</p>

## How it works

The figure reads left to right.

**1. Real data in.** The city is built from public and survey data (green is
empirical, yellow is survey-derived):

| Source | What it gives the simulation |
|---|---|
| OpenStreetMap | the street network everyone travels on |
| LODES | real home → work commutes, so people live and work where people actually do |
| ACS | who lives where: age, income, household, car ownership |
| National POI catalog | real restaurants, cafés, gyms, parks, museums and music venues |
| Survey (473 people) | how people feel about platforms, their personalities and leisure preferences |

**2. People with their own way of deciding.** Each agent has a home, a job, an
income, a value of time and hidden tastes (say, a soft spot for pizza). After
work they decide whether to go out and where. Not everyone decides the same
way: some weigh every option carefully, some settle for good enough, some
avoid regret, and some just repeat what they did last time. Trips happen on
the real street network by foot, bike-share, car or taxi, and each one costs
time, money and CO₂.

**3. A recommender you can tune.** The recommender scores venues on six
things: rating, number of reviews, relevance, proximity, popularity and
personalization. You set how much each one counts. Turn up proximity and you
get something like a maps app; turn up popularity and you get a "trending"
feed. Agents decide whether to trust each suggestion, leave feedback after
the visit, and that feedback updates ratings, footfall and what the
recommender knows about them. Tomorrow's ranking depends on today's trips.

**4. Outcomes out.** Every run is compared against a matched control where
nobody uses a recommender, so every difference you see is caused by the
platform. You get:

- **Footfall concentration**: do visits pile into a handful of venues?
- **Visitor concentration**: which venues and neighborhoods win or lose customers
- **Net utility by income**: who actually came out ahead, and who paid for it
- **Trip costs**: extra distance, travel time and emissions
- **Activity benefit over time**: whether the recommender learns people's tastes

A second control, an "oracle" that knows everyone's true preferences, shows
the best any recommender could do.

## Questions you can ask with it

- Does a popularity-heavy recommender funnel everyone into the same few places?
- If a platform favors nearby venues, does it cut driving and emissions? What
  does that cost people in enjoyment?
- Do the benefits of a recommender reach low-income travelers, or mostly
  people who can afford longer trips?
- Can a welfare filter, one that only shows options likely to leave someone
  better off, keep the upside without the downsides?

## Cities

Eight US metro areas: **New York, Los Angeles, Chicago, Houston, Washington DC,
Miami, Seattle** and the **San Francisco Bay Area**.

Venues and workplaces sit in the principal city (Manhattan for New York; San
Francisco and Oakland for the Bay Area). Homes extend across the surrounding
counties that send 90% of commuters into it.

## Quick start

The easiest way in is the [live site](https://airecsim.cusp.nyu.edu/). To run
it yourself:

### Run a small demo

No data download or database needed.

```bash
git clone https://github.com/nahvats/GeoRecSim.git
cd GeoRecSim
pip install -e .
python data/make_demo_personas.py
python scripts/run_demo.py
```

This runs 50 made-up people on a small piece of Manhattan for 3 days. It's
for checking that everything works. The people are invented, so the numbers
don't mean anything.

### Run the web app locally

```bash
pip install -e .[server]
cd backend && uvicorn main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend && npm install && npm run dev
```

Then open <http://localhost:5173>. More options, including a single-server
setup and the HTTP API, are in [docs/running-the-app.md](docs/running-the-app.md).

### Use it from Python

The simulation engine (`welfare_rs`) works on its own, with no web stack:

```python
from welfare_rs import Simulation, build_metro_network, params

net = build_metro_network("miami")          # downloaded once, then cached
sim = Simulation(
    num_agents=80,
    seed=42,
    road_network=net,
    persona_csv_path=params.NYC_PERSONA_CSV_PATH,
)
sim.run_days(3)
print(sim.summarize())
```

## About the data

The repo contains code, not data. Census files are large and re-downloadable,
the venue catalog is licensed, and the survey responses are IRB-protected and
stay on university storage. Scripts in [`db/`](db/) rebuild every piece, and
[docs/data.md](docs/data.md) explains where each input comes from and how to
build it.

## What's in the repo

| Folder | What it is |
|---|---|
| [`welfare_rs/`](welfare_rs/) | the simulation engine, installable with `pip install -e .` |
| [`backend/`](backend/) | FastAPI service that runs simulations and streams results to the map |
| [`frontend/`](frontend/) | React + TypeScript web app (deck.gl and MapLibre) |
| [`db/`](db/) | database schema and scripts that build the population and venue tables |
| [`data/`](data/) | persona builders and the demo population |
| [`scripts/`](scripts/) | the demo, network prefetching, study runners and checks |
| [`gating_study/`](gating_study/) | experiments on welfare filters, with results |
| [`docs/`](docs/) | longer write-ups (below) |

## Documentation

- [Running the app](docs/running-the-app.md): install, local setup, HTTP API, running many studies at once
- [The recommender](docs/recommender.md): every knob, the oracle control, welfare filters, hidden tastes
- [Data](docs/data.md): cities, commutes, venues, and how to rebuild everything
- [The agent population](docs/acs-population.md): how census and survey data become agents
- [What the survey covers](docs/survey-coverage.md): which agent traits are measured and which are not
- [Getting around](docs/multimodal.md): walking, bike-share, car and taxi

## Team

GeoRecSim is built at New York University by
**Nahush Shrivatsa**, **Ekin Uğurel** and **Takahiro Yabe**.

It is a research prototype and still under active development. If something
breaks or you'd like to use it in your own work, please
[open an issue](https://github.com/nahvats/GeoRecSim/issues).

## Citation

A paper describing GeoRecSim is in preparation. Until then, please cite:

```bibtex
@misc{shrivatsa2026georecsim,
  title        = {GeoRecSim: A Multi-Agent Testbed for Recommender System Design},
  author       = {Shrivatsa, Nahush and Ugurel, Ekin and Yabe, Takahiro},
  year         = {2026},
  howpublished = {\url{https://github.com/nahvats/GeoRecSim}}
}
```
