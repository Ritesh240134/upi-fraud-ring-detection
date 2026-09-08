# Real-Time Detection of UPI Fraud Rings using Graph Neural Networks

Fraud rings do not look suspicious one transaction at a time. A mule account
receiving ₹9,000 looks like anyone else; twelve accounts moving money in a
ring within a ten-minute window does not. This project models UPI activity as
a temporal heterogeneous graph and detects the ring rather than the payment.

Everything here runs on CPU, on synthetic data, in about six minutes end to
end.

---

## What it does

1. **Simulates** a UPI-like event stream — transfers, logins, beneficiary
   additions and merchant collections — with six fraud typologies injected
   (smurfing, mule-farm bursts, fan-out layering, circular round-trips,
   merchant collusion, social-engineering harvests).
2. **Builds** a heterogeneous graph over accounts, devices, VPAs and
   merchants, with a timestamp on every edge.
3. **Trains** a temporal heterogeneous GNN that scores each account's
   probability of belonging to a fraud ring.
4. **Extracts rings** by sliding a time window over high-risk transfers and
   pulling out connected components.
5. **Serves** the scores through a FastAPI service, a web console at
   `/dashboard`, and a Streamlit page for quick analysis.
6. **Detects live**: events streamed in over HTTP update the graph and
   surface rings as they form, scoring accounts never seen in training.

---

## Results

All figures below come from `models/artifacts/*_eval_metrics.json`, written by
the evaluation scripts. Nothing is hand-entered — regenerate them with the
commands in [Running it](#running-it).

### Account classification, held-out test split

The test split holds **75 accounts, 11 of them fraudulent**. The decision
threshold is chosen on the validation split and not re-tuned on test.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| GraphSAGE baseline (homogeneous) | 0.853 | 0.500 | 0.364 | 0.421 | 0.750 | n/a |
| Temporal hetero GNN, time withheld | 0.867 | 0.545 | 0.545 | 0.545 | 0.847 | 0.668 |
| **Temporal hetero GNN** | **0.867** | **0.538** | **0.636** | **0.583** | **0.861** | **0.704** |

| Model | TN | FP | FN | TP | threshold |
|---|---|---|---|---|---|
| GraphSAGE baseline | 60 | 4 | 7 | 4 | 0.77 |
| Temporal hetero GNN, time withheld | 59 | 5 | 5 | 6 | 0.36 |
| Temporal hetero GNN | 58 | 6 | 4 | 7 | 0.38 |

**Read these numbers carefully.** With 11 positives in the test split, one
account moving between columns changes F1 by roughly 0.05. The gap between
the temporal model and its time-withheld ablation is a single account
(7 true positives vs 6). The ordering is consistent across ROC-AUC, PR-AUC
and F1, and the ablation is a genuine control — same architecture, same seed,
same split, only the timestamps withheld — but a one-account margin on a
75-account split is suggestive, not conclusive. Treat it as "time does not
hurt and probably helps", not as a demonstrated win.

The jump from the baseline to the heterogeneous models (ROC-AUC 0.750 →
0.847) is the larger and more trustworthy effect.

### Ring detection

Running the detector over the full graph:

- **12 rings** detected
- **73 accounts** inside them, of which **67 are genuinely fraudulent (91.8%)**
- those 67 cover **97.1% of the 69 fraud accounts** in the dataset
- ring sizes 3–19, time spans 1.0–9.8 minutes

This is the number to be most careful about: ring extraction scores *every*
account in the graph, including the 350 the model trained on. It is an
operational demonstration of the end-to-end pipeline, **not a held-out
metric**. The honest generalisation numbers are the test-split table above.

---

## Architecture

```
data/synthetic/upi_events.csv          5,086 events, 500 accounts, 86 fraud events
        |
        +-- graph/prepare_graph.py         -> homogeneous account graph (baseline)
        |
        +-- graph/prepare_hetero_graph.py  -> temporal heterogeneous graph
                    |
                    |   account  (13 behavioural features)
                    |   device   (4 activity features)
                    |   vpa      (4 activity features)
                    |   merchant (4 activity features)
                    |
                    |   9 relations, every one carrying edge_time:
                    |     account -> transfers        -> account
                    |     account -> rev_transfers    -> account
                    |     account -> adds_beneficiary -> account
                    |     account -> logs_in          -> device
                    |     device  -> rev_logs_in      -> account
                    |     account -> owns             -> vpa
                    |     vpa     -> rev_owns         -> account
                    |     account -> collects_at      -> merchant
                    |     merchant-> rev_collects_at  -> account
                    v
        models/temporal_hetero_gnn.py      154,483 parameters
                    |
                    +-- detection/gnn_ring_detector.py -> rings + account scores
                    +-- backend/main.py                -> FastAPI serving
                    |        +-- backend/dashboard_api.py -> /api/* for the console
                    |        +-- backend/stream_service.py-> live detection
                    |        +-- frontend/               -> web console at /dashboard
                    +-- dashboard/app.py               -> Streamlit page
```

### How time enters the model

Each relation is convolved with `TemporalSAGEConv`:

```
out_i = W_self · x_i  +  W_neigh · mean_j( gate(Δt_ij) · x_j )
```

`Δt_ij` is how long before the most recent edge of that relation the
interaction occurred. It is passed through a Time2Vec-style encoder (one
linear component plus learned-frequency periodic components) and then a
learned gate into (0, 1) that scales the neighbour's message. Recent activity
therefore dominates an account's representation, which is what separates a
mule-farm burst from ordinary long-running activity.

Withholding `edge_time` reduces every conv to plain mean aggregation — that is
exactly the `--no-time` ablation above, so the comparison isolates the
temporal machinery and nothing else.

---

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Then, in order:

```bash
python -m simulator.upi_simulator            # regenerate the event stream
python -m graph.create_labels                # account-level fraud labels
python -m graph.prepare_graph                # homogeneous baseline graph
python -m graph.prepare_hetero_graph         # temporal heterogeneous graph
python -m training.train_gnn                 # GraphSAGE baseline
python -m training.evaluate_gnn
python -m training.train_temporal_hetero     # temporal hetero GNN
python -m training.evaluate_temporal_hetero
python -m detection.ring_detector            # rule-based reference rings
python -m detection.explainer                # per-ring event traces
python -m detection.gnn_ring_detector        # GNN-driven rings + risk scores
python -m pytest tests/ -q
```

The time ablation:

```bash
python -m training.train_temporal_hetero --no-time
python -m training.evaluate_temporal_hetero --no-time
```

Serving:

```bash
uvicorn backend.main:app --reload
```

| URL | What it is |
|---|---|
| `http://127.0.0.1:8000/dashboard` | Web console — the main interface |
| `http://127.0.0.1:8000/docs` | Interactive OpenAPI docs |

And the Streamlit page, which stays as a quick analyst view:

```bash
streamlit run dashboard/app.py               # http://localhost:8501
```

### In VS Code

Open the project folder, accept the recommended Python extensions, and pick
the interpreter at `.venv\Scripts\python.exe`
(`Ctrl+Shift+P` → *Python: Select Interpreter*).

- **Ctrl+Shift+B** runs the whole pipeline end to end.
- **F5** → *API + web console* starts the server with the debugger attached.
- The **Testing** panel runs the test suite.
- Individual stages are under `Ctrl+Shift+P` → *Tasks: Run Task*.

One gotcha worth knowing: these modules import each other as a package, so
running a file with the plain ▶ button fails with
`ModuleNotFoundError: No module named 'graph'`. `.vscode/settings.json` sets
`PYTHONPATH` to the project root to prevent that; from a terminal, always use
`python -m graph.prepare_hetero_graph` rather than
`python graph/prepare_hetero_graph.py`.

---

## Live streaming detection

The batch pipeline scores a frozen graph. The streaming detector keeps it
alive: events arrive over HTTP, the graph is rebuilt, accounts are rescored,
and rings are extracted from a sliding window — no retraining, and accounts
the model has never seen are scored on arrival.

Start the API, then replay the event stream into it:

```bash
python -m streaming.replay --reset --tail 300 --speed 5000
```

Watch it at `http://127.0.0.1:8000/dashboard#live`. A representative run:

```
Warming detector with 4,786 historical events...
Baseline: 500 accounts in 152.9 ms
  [  240/300] rings 0 -> 2  (+2)
  [  270/300] rings 2 -> 7  (+5)
  [  300/300] rings 7 -> 12 (+5)

Accounts known   : 500
Flagged accounts : 72
Active rings     : 12
Last rescore     : 355.2 ms
```

Rings form as the fraud burst arrives, and the stream converges on exactly
the same 12 rings and 72 flagged accounts the batch pipeline finds — the two
paths share one graph-construction function, so they cannot drift apart.

### Why no retraining is needed

`TemporalHeteroGNN` has no node-ID embedding tables: every node type is a
`nn.Linear` over its **features**, followed by neighbour aggregation. The
architecture is inductive, so a brand-new account is scoreable the moment it
has features and edges. Two things make that work in practice:

- **The feature scaler is persisted.** `data/processed/feature_scaler.json`
  stores the training mean and standard deviation per node type. Live
  features are standardised with those, never with statistics refitted on
  live traffic — refitting would put the inputs on a different scale from
  the ones the model was trained on.
- **The time scale is fixed.** The same file stores the training time span.
  Without it, every arriving event would stretch the min/max span and
  retroactively rescale the time on every historical edge, moving unrelated
  accounts' scores. (This one was a real bug: ingesting ten events collapsed
  ring detection from 12 rings to 2.)

Existing scores do still move when the clock genuinely advances — that is the
temporal gate decaying old evidence, which is the point of the model.

### Streaming endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/stream/events` | Ingest a batch of events |
| POST | `/api/stream/event` | Ingest one event |
| GET | `/api/stream/state` | Counters, alerts, top accounts |
| GET | `/api/stream/rings` | Rings in the live window |
| POST | `/api/stream/rescore` | Force a rescore |
| POST | `/api/stream/reset` | Return to the seeded baseline |

Rescoring is batched (every 25 events by default) — a forward pass per event
would dominate the cost without changing the answer. Events with an
unparseable timestamp are rejected and reported rather than dropped.

---

## Web console

`/dashboard` is a single-page console served by the API itself. It has no
build step and no external dependencies — the charts and the ring graph are
hand-rolled SVG, so it works offline and there is nothing to `npm install`.

- **Overview** — headline counters, ring-detection quality, event mix and the
  injected fraud typologies.
- **Fraud rings** — every detected ring, with a drill-down showing the money
  flow as a directed graph, the member accounts coloured by risk, matched
  ground-truth rings, and the event timeline for the window.
- **Accounts** — the score distribution with the decision threshold marked,
  filter and search over all 500 accounts, and a single-account lookup.
- **Live scoring** — post a transaction and see the decision, the score for
  each endpoint, the rule-layer contribution and the reasons behind it.
  Two preset buttons demonstrate a known ring pair and unseen accounts.
- **Model** — the measured test metrics as a grouped bar chart, a confusion
  matrix per model, and the caveat about the small test split.

Dark and light themes, keyed to a toggle and remembered in `localStorage`.
The layout collapses to an icon rail on narrow screens.

Everything on the page is read from the API at load time. The number of
accounts, rings and every metric come from `/api/overview`, `/api/metrics`
and `/api/rings` — none of it is hard-coded in the frontend.

Everything is seeded (`SEED = 42`) and CPU-only. Two runs of the simulator
produce byte-identical CSVs.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/rings` | Rule-based detected rings |
| GET | `/rings/gnn` | GNN-detected rings |
| GET | `/rings/{ring_id}` | One rule-based ring |
| GET | `/explanations` | Per-ring event traces |
| GET | `/explanations/{ring_id}` | Events for one ring |
| GET | `/events/summary` | Event counts by type |
| GET | `/accounts/{account_id}/risk` | Model score for one account |
| GET | `/model/info` | Which model is loaded, and its threshold |
| POST | `/score` | Score a transaction |
| GET | `/dashboard` | Web console |

Backing the console (additive; the endpoints above kept their contracts):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/overview` | Headline counters and event mix |
| GET | `/api/metrics` | Measured metrics for all three models |
| GET | `/api/rings` | Detected rings |
| GET | `/api/rings/{ring_id}` | One ring with members, edges and timeline |
| GET | `/api/accounts` | Paged, filterable, searchable account list |
| GET | `/api/accounts-summary` | Score histogram and risk-level counts |

`POST /score` returns the model's view of both endpoints plus the rule-layer
contribution, so a decision can be audited:

```json
{
  "source_account": "ACC_00326",
  "target_account": "ACC_00109",
  "amount": 75000.0,
  "risk_score": 1.0,
  "risk_level": "HIGH",
  "fraud_probability": 1.0,
  "source_account_score": 1.0,
  "target_account_score": 1.0,
  "rule_layer_score": 0.25,
  "reasons": [
    "elevated_transaction_amount",
    "source_account_flagged_by_gnn",
    "target_account_flagged_by_gnn",
    "both_endpoints_in_suspected_ring"
  ],
  "model_status": "temporal_hetero_gnn"
}
```

Accounts the model has never seen fall back to the rule layer and say so via
`model_status`, rather than erroring. If the model artifacts are missing
entirely, the API still starts and every endpoint still answers — `/score`
reports `rule_layer_model_unavailable`.

---

## Layout

```
backend/          FastAPI service, model serving, console API, streaming
frontend/         Web console (index.html, styles.css, app.js)
streaming/        Event replay driver for live detection
dashboard/        Streamlit page
data/synthetic/   Generated event stream
data/processed/   Graphs, labels, risk scores, detected rings
detection/        Rule-based and GNN-driven ring detection, explainer
graph/            Graph construction and feature engineering
models/           Model definitions
models/artifacts/ Trained checkpoints and measured metrics
simulator/        Synthetic UPI event generator
training/         Training and evaluation scripts
tests/            72 tests
```

### Superseded files

Three modules are kept for reference but are no longer part of the pipeline;
each carries a header saying what replaced it:

- `graph/build_graph.py` and `graph/prepare_gnn_data.py` → replaced by
  `graph/prepare_hetero_graph.py`
- `models/hetero_gnn.py` → replaced by `models/temporal_hetero_gnn.py`
- `backend/realtime_scorer.py` → now a thin shim re-exporting
  `backend.main:app`

---

## Notes on correctness

Four issues were found and fixed while completing this project. They are
recorded here because each one produced misleading results:

1. **Label leakage in the baseline features.** `graph/prepare_graph.py` used a
   per-account count of `is_fraud` events as node feature 7 — the label,
   fed in as an input. Training loss fell to 0.0039 and the model looked
   near-perfect. Removing it drops the baseline to ROC-AUC 0.750, which is
   the real number. `tests/test_gnn.py::test_features_do_not_leak_the_label`
   guards against a reintroduction.

2. **Evaluation on training data.** The baseline trained on all 500 accounts
   and evaluated on all 500. There are now stratified train/val/test masks
   (70/15/15), the threshold is picked on validation, and test is touched
   only by the evaluation scripts.

3. **A "temporal" model that ignored time.** `TemporalHeteroGNN` accepted a
   graph carrying `edge_time` and never read it — it was plain heterogeneous
   GraphSAGE. Time is now consumed by `TemporalSAGEConv`, and two tests
   assert that both supplying timestamps and *changing their values* alters
   the output.

4. **A ring detector that could never fire.** It built one graph from every
   suspicious edge in the dataset, then discarded components spanning more
   than ten minutes. Since all suspicious edges collapse into one component
   spanning the whole dataset, every candidate was rejected and the detector
   always returned zero rings. Components are now built inside a sliding
   window instead of filtered afterwards.

A fifth, smaller issue: the simulator seeded `random.Random(seed)` but drew
event IDs from `uuid.uuid4()`, which reads OS entropy and ignores the seed, so
no run was reproducible. IDs now come from a dedicated seeded stream, kept
separate from the main one so that generating them cannot shift the event
sequence.

---

## Limitations

- **Synthetic data.** Typologies are injected by a generator that knows what
  it is drawing. Real UPI fraud is messier and the labels are noisier.
- **Small evaluation set.** 500 accounts, 69 fraudulent, 75 in the test split.
  Confidence intervals on every metric above are wide.
- **`POST /score` is a lookup.** That endpoint answers from a table
  precomputed at startup, so an account missing from it falls back to the
  rule layer. The streaming path does score unseen accounts; the two are not
  yet wired together.
- **Ring metrics are in-sample**, as flagged above.
- **Rescoring is whole-graph.** Live detection rebuilds the graph and runs a
  full forward pass. At this size that is ~200-350 ms, which is genuinely
  real-time; a national payment network would need a graph store with
  neighbour sampling instead.

---

All data in this repository is synthetic. Nothing here is derived from real
UPI transactions.
