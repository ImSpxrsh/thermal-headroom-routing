# Thermal-Headroom Routing

Code, data and paper for *Thermal-Headroom Routing for Efficient Robot Perception in Extreme Heat*.

Authors: Sparsh Roy (MIT and Hopewell Valley Central HS, corresponding author, sparshr@mit.edu),
Arhaan Shah (Paradise Valley HS), Kevin Zhou (Hopewell Valley Central HS), and three more Paradise Valley HS authors to be added.

We are submitting this to the 1st Data and Model Efficiency for Scalable AI Workshop (Efficient AI)
at IEEE BigData 2026 in Phoenix. The PDF is [`paper/paper.pdf`](paper/paper.pdf) and the source is
[`paper/paper.tex`](paper/paper.tex).

The project page is [`index.html`](index.html), a single static file. With GitHub Pages turned on for `main` (root folder), it is served at https://imspxrsh.github.io/thermal-headroom-routing/.

## What this is about

Put a Jetson-class board in a sealed box on a robot in the Arizona desert and run the biggest
detector on every frame. Within about a minute the chip hits its thermal limit, the governor
down-clocks it, and every model gets slower, the small ones included. Adaptive inference routers
pick a model size and resolution per frame, but they only react to latency once it has gone up.
In the heat that is too late, because the router's own earlier choices set up the slowdown.

THR gives the router a temperature forecast. An RC thermal model predicts, in closed form, how hot
the chip will get if the router keeps using a given configuration, and THR skips any configuration
that would push it past the throttle threshold. Each decision takes about 7 µs.

## Results

These come from a closed-loop simulation that uses real camera traces and real profiled latencies:
7 traces, 8 ambient temperatures, 20 seeds. Significance is Holm-corrected across the 7 traces.

With the deadline at the big model's median latency:

| | 45 °C (Phoenix summer) | 55 °C |
|---|---|---|
| Share of mission throttled, belief router vs. THR | 11 to 13 % vs. 0 % | 32 to 38 % vs. 0 % |
| Drop in deadline-miss rate vs. belief router | 3.9 to 5.4 pp, significant on 7 of 7 | 6.7 to 7.5 pp, significant on 7 of 7 |
| Energy per on-time frame vs. belief router | 12 % lower | 13 % lower |
| Longest blackout when always running the big model | 12.5 s | 90.7 s |

At 35 °C and below THR makes exactly the same choices as the baseline.

We compared THR against baselines picked to beat it:

- A thermostat that bars the big model above 88 °C also never throttles on the chip it was set for.
  THR misses 1.3 to 4.4 pp fewer deadlines than that setting on every trace. Sweeping the
  threshold traces a trade-off curve, and THR lands on it.
- Every temperature-aware policy (thermostat, duty caps, THR) sits on the same accuracy-timeliness
  curve, set by the share of big-model frames. A control given THR's own share matches its miss
  rate, so at one temperature on a known chip the forecast adds little.
- A cap tuned at 45 °C throttles 16 to 31 % of the time at 50 to 55 °C, where THR misses 3.1 to
  6.6 pp fewer deadlines.
- When the real chip runs 20 % hotter than assumed, the thermostat throttles up to 29 % of the time
  at 55 °C. THR stayed at 0 % on all five chip variants we tried. This is THR's distinct gain.
- With 25 % deadline slack, the small models still finish on a throttled chip, so throttling costs
  little. There the belief router has higher on-time utility than THR on 6 of 7 traces.

When camera faults also add compute load, combining THR with fault-aware routing beats the belief
router by 5.8 to 13.3 pp at 45 and 55 °C. Giving THR a wrong thermal model never brought throttling
back; it only shifts THR along the accuracy-timeliness curve.

Accuracy in this project means agreement with the largest model on the same frame, which is a
pseudo ground truth, so read accuracy numbers as an internal check. The thermal parameters are
values we chose for a sealed, sun-loaded enclosure, not measurements. The paper's limitations
section has the full list.

## Reproducing the results

You do not need a GPU or the raw datasets. Everything runs from the cached per-track outputs in
`outputs/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib pyyaml python-dotenv
python src/run_extreme_env.py          # about 8 minutes; writes outputs/extreme_env/extreme_env.json
python src/make_extreme_figures.py     # writes paper/figures/*.pdf and paper/tables/*.tex
cd paper && latexmk -pdf paper.tex
```

| File | What it does |
|---|---|
| `src/thermal.py` | RC thermal plant, online compute belief, the THR policy, the closed-loop runner, blackout metrics |
| `src/run_extreme_env.py` | Experiments E1 to E9: bursts, ambient sweep with thermostat and cap baselines, heat plus camera faults, wrong thermal model, deadline sweep, different chips, THR and thermostat settings, fixed-cap curve |
| `src/make_extreme_figures.py` | Builds every figure and table in the paper from the JSON, so no number is copied by hand |
| `config.yaml`, `thermal:` section | Plant parameters, set before any run |
| `src/jetson_harness.py` | Logs real latency plus `tegrastats` temperature and power on a Jetson, to replace the chosen parameters with measured ones |

## Where the substrate comes from

The model frontier, the camera-fault belief estimators, the dataset tracks (TartanDrive, RADIATE,
WoodScape) and the cached per-frame outputs come from *Belief-Space Perception Routing* by Roy,
Aggarwal and Yin ([arXiv:2608.00322](https://arxiv.org/abs/2608.00322)). This repository started as
a copy of that codebase, which is the `init` commit. The original README, paper and run guides are
in [`docs/bspr-substrate/`](docs/bspr-substrate/), 

## License

MIT. See [LICENSE](LICENSE).
