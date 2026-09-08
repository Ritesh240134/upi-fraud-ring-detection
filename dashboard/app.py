"""
Streamlit dashboard for UPI fraud-ring detection.

Run:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent.parent

PROCESSED = BASE / "data" / "processed"
ARTIFACTS = BASE / "models" / "artifacts"

EVENTS_FILE = BASE / "data" / "synthetic" / "upi_events.csv"
LABELS_FILE = PROCESSED / "account_labels.csv"
RINGS_FILE = PROCESSED / "detected_rings.csv"
GNN_RINGS_FILE = PROCESSED / "gnn_detected_rings.csv"
SCORES_FILE = PROCESSED / "account_risk_scores.csv"

BASELINE_METRICS = ARTIFACTS / "graphsage_eval_metrics.json"
TEMPORAL_METRICS = ARTIFACTS / "temporal_hetero_eval_metrics.json"
ABLATION_METRICS = ARTIFACTS / "temporal_hetero_eval_metrics_no_time.json"


st.set_page_config(
    page_title="UPI Fraud Ring Detection",
    layout="wide",
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


events = load_csv(EVENTS_FILE)
labels = load_csv(LABELS_FILE)
rings = load_csv(RINGS_FILE)
gnn_rings = load_csv(GNN_RINGS_FILE)
scores = load_csv(SCORES_FILE)

baseline_metrics = load_json(BASELINE_METRICS)
temporal_metrics = load_json(TEMPORAL_METRICS)
ablation_metrics = load_json(ABLATION_METRICS)


# ============================================================
# HEADER
# ============================================================

st.title("Real-Time UPI Fraud Ring Detection")
st.caption(
    "Temporal heterogeneous graph neural network over synthetic UPI events"
)

column_1, column_2, column_3, column_4, column_5 = st.columns(5)

column_1.metric(
    "Accounts",
    len(labels) if not labels.empty else 0,
)

column_2.metric(
    "Fraud accounts",
    int(labels["label"].sum())
    if not labels.empty and "label" in labels.columns
    else 0,
)

column_3.metric(
    "GNN-detected rings",
    len(gnn_rings) if not gnn_rings.empty else 0,
)

column_4.metric(
    "Events",
    len(events) if not events.empty else 0,
)

column_5.metric(
    "Test ROC-AUC",
    f"{temporal_metrics['test']['roc_auc']:.3f}"
    if temporal_metrics
    else "n/a",
)

st.divider()


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("Model performance (held-out test split)")

st.caption(
    "All figures come from `models/artifacts/*_eval_metrics.json`, "
    "produced by the evaluation scripts. Nothing here is hand-written."
)


def metric_row(name: str, payload: dict) -> dict:

    if not payload:
        return {}

    test = payload["test"]

    return {
        "Model": name,
        "Accuracy": test["accuracy"],
        "Precision": test["precision"],
        "Recall": test["recall"],
        "F1": test["f1"],
        "ROC-AUC": test["roc_auc"],
    }


comparison = [
    row
    for row in (
        metric_row("GraphSAGE baseline", baseline_metrics),
        metric_row("Temporal hetero GNN (no time)", ablation_metrics),
        metric_row("Temporal hetero GNN", temporal_metrics),
    )
    if row
]

if comparison:

    table = pd.DataFrame(comparison).set_index("Model")

    st.dataframe(
        table.style.format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "The test split holds 75 accounts, 11 of them fraudulent. "
        "Differences of one or two accounts move these numbers noticeably."
    )

else:
    st.info(
        "No evaluation metrics yet. Run the training and evaluation scripts."
    )


st.divider()


# ============================================================
# GNN-DETECTED RINGS
# ============================================================

st.subheader("GNN-detected fraud rings")

if gnn_rings.empty:
    st.info(
        "No GNN rings available. Run `python -m detection.gnn_ring_detector`."
    )

else:

    flagged = int(gnn_rings["ring_size"].sum())
    correct = int(gnn_rings["true_fraud_accounts"].sum())

    left, right = st.columns(2)

    left.metric("Accounts inside detected rings", flagged)

    right.metric(
        "Genuinely fraudulent",
        f"{correct} ({correct / flagged:.1%})" if flagged else "0",
    )

    st.dataframe(
        gnn_rings[
            [
                "detected_ring_id",
                "ring_size",
                "suspicious_edges",
                "time_span_minutes",
                "ring_risk_score",
                "true_fraud_accounts",
                "total_amount",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox(
        "Inspect a ring",
        gnn_rings["detected_ring_id"].tolist(),
    )

    ring = gnn_rings[
        gnn_rings["detected_ring_id"] == selected
    ].iloc[0]

    members = str(ring["accounts"]).split(",")

    st.write(
        f"**{selected}** - {ring['ring_size']} accounts over "
        f"{ring['time_span_minutes']} minutes, "
        f"risk {ring['ring_risk_score']}"
    )

    if not events.empty:

        ring_events = events[
            events["source_account"].astype(str).isin(members)
            | events["target_account"].astype(str).isin(members)
        ]

        st.dataframe(
            ring_events.sort_values("timestamp").head(100),
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ============================================================
# ACCOUNT RISK
# ============================================================

st.subheader("Account risk scores")

if scores.empty:
    st.info(
        "No account scores yet. "
        "Run `python -m detection.gnn_ring_detector`."
    )

else:

    left, right = st.columns([2, 3])

    with left:

        st.write("Risk level distribution")

        # Passed as an explicit two-column frame with named x and y. Handing
        # st.bar_chart a bare value_counts Series rendered an empty chart with
        # a nonsensical axis range.
        distribution = (
            scores["risk_level"]
            .value_counts()
            .rename_axis("risk_level")
            .reset_index(name="accounts")
        )

        st.bar_chart(
            distribution,
            x="risk_level",
            y="accounts",
        )

    with right:

        st.write("Highest-risk accounts")

        st.dataframe(
            scores.sort_values(
                "fraud_probability",
                ascending=False,
            ).head(20),
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ============================================================
# RULE-BASED RINGS AND EVENT STATISTICS
# ============================================================

left, right = st.columns(2)

with left:

    st.subheader("Rule-based rings")

    st.caption(
        "Connected components of events already flagged `is_fraud`. "
        "A ground-truth reference, not a predictor."
    )

    if rings.empty:
        st.info("No rule-based rings available.")
    else:
        st.dataframe(
            rings,
            use_container_width=True,
            hide_index=True,
        )

with right:

    st.subheader("Event statistics")

    if not events.empty and "event_type" in events.columns:

        event_counts = (
            events["event_type"]
            .value_counts()
            .rename_axis("event_type")
            .reset_index(name="events")
        )

        st.bar_chart(
            event_counts,
            x="event_type",
            y="events",
        )

        if "typology" in events.columns:

            st.write("Fraud typologies")

            st.dataframe(
                events[events["typology"] != "NORMAL"]["typology"]
                .value_counts()
                .rename("events"),
                use_container_width=True,
            )

    else:
        st.info("No event data available.")


st.divider()

st.caption(
    "Synthetic data only. Every metric shown is reproducible with the "
    "commands in README.md."
)
