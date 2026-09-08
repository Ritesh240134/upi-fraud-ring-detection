"""
Build the temporal heterogeneous UPI graph.

Node types
----------
    account, device, vpa, merchant

Relations (each carries `edge_time`)
------------------------------------
    account -> transfers        -> account   (TRANSFER)
    account -> rev_transfers    -> account   (reverse of the above)
    account -> adds_beneficiary -> account   (BENEFICIARY_ADD)
    account -> logs_in          -> device
    device  -> rev_logs_in      -> account
    account -> owns             -> vpa
    vpa     -> rev_owns         -> account
    account -> collects_at      -> merchant  (MERCHANT_COLLECTION)
    merchant-> rev_collects_at  -> account

`edge_time` is the event time normalised to [0, 1] across the dataset span.
The temporal model converts it into a per-edge recency used to gate messages,
so a stale login and a login five seconds ago are not treated identically.

Account features are the same leak-free behavioural features used by the
homogeneous baseline - see graph/prepare_graph.py. No column derived from
`is_fraud`, `ring_id` or `typology` is ever used as an input feature.

`build_hetero_graph` is the single construction path, shared by this script
and by the live streaming detector, so an account is described identically
whether it is scored in a batch run or on arrival.

Outputs:
    data/processed/hetero_graph_data.pt
    data/processed/feature_scaler.json
"""

from pathlib import Path

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from graph.prepare_graph import (
    apply_scaler,
    build_account_index,
    build_features,
    build_labels,
    build_masks,
    fit_scaler,
    load_events,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"
SCALER_FILE = ROOT / "data" / "processed" / "feature_scaler.json"

ENTITY_TYPES = ("device", "vpa", "merchant")

# Key under which the training time scale is stored in the scaler file.
TIME_SPAN_KEY = "__time_span_seconds__"


# ============================================================
# TIME NORMALISATION
# ============================================================

def normalised_time(df: pd.DataFrame, span_seconds: float = None):
    """
    Scale timestamps into time units the model was trained on.

    Returns (series, span_seconds_used).

    `span_seconds` fixes the scale. Training passes None, which measures the
    span of the training data and returns it so it can be stored. Serving
    passes the stored value.

    This matters for streaming. If the scale were re-derived from whatever is
    in the buffer, every newly ingested event would stretch the span and
    retroactively rescale the time on every historical edge - so old accounts
    would change score merely because unrelated new traffic arrived. Fixing
    the scale keeps history stable and lets new events simply extend it.
    """

    # Convert to seconds explicitly. `astype("int64")` returns whatever
    # resolution pandas happens to be storing (nanoseconds on some versions,
    # microseconds on others), so dividing by a hard-coded 1e9 gives a value
    # whose unit depends on the pandas build - which would make a stored time
    # span meaningless across environments.
    seconds = pd.Series(
        df["timestamp"]
        .to_numpy(dtype="datetime64[s]")
        .astype("int64")
        .astype(float),
        index=df.index,
    )

    seconds = seconds.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    start = seconds.min()

    span = span_seconds

    if span is None:
        span = seconds.max() - start

    if span is None or not np.isfinite(span) or span <= 0:
        return (
            pd.Series(
                np.zeros(len(df)),
                index=df.index,
                dtype=float,
            ),
            None,
        )

    return ((seconds - start) / span).fillna(0.0), float(span)


# ============================================================
# NODE INDEXES
# ============================================================

def build_entity_index(values: pd.Series) -> dict:

    unique = sorted(
        value
        for value in values.dropna().astype(str).unique()
        if value.strip() and value.lower() != "nan"
    )

    return {value: index for index, value in enumerate(unique)}


# ============================================================
# ENTITY FEATURES
# ============================================================

def entity_features(
    df: pd.DataFrame,
    column: str,
    index_map: dict,
    time: pd.Series,
) -> np.ndarray:
    """
    Raw (unscaled) activity profile for a non-account node type:

        [ log(event count),
          log(distinct accounts),
          log1p(mean amount),
          mean normalised event time ]
    """

    if not index_map:
        return np.zeros((0, 4), dtype=np.float32)

    keys = pd.Index(list(index_map.keys()))

    values = df[column].astype(str)

    frame = pd.DataFrame(
        {
            "key": values,
            "account": df["source_account"].astype(str),
            "amount": df["amount"],
            "time": time,
        }
    )

    frame = frame[frame["key"].isin(index_map)]

    grouped = frame.groupby("key")

    def reindex(series: pd.Series) -> np.ndarray:
        return (
            series.reindex(keys)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    features = np.column_stack(
        [
            np.log1p(reindex(grouped.size())),
            np.log1p(reindex(grouped["account"].nunique())),
            np.log1p(reindex(grouped["amount"].mean())),
            reindex(grouped["time"].mean()),
        ]
    ).astype(np.float32)

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def scale(features: np.ndarray, node_type: str, scalers: dict) -> torch.Tensor:
    """
    Apply the stored scaler for a node type, fitting it first if this is the
    training run (scalers is being built rather than reused).
    """

    if features.shape[0] == 0:
        return torch.zeros(features.shape, dtype=torch.float)

    if node_type not in scalers:
        scalers[node_type] = fit_scaler(features)

    return torch.tensor(
        apply_scaler(features, scalers[node_type]),
        dtype=torch.float,
    )


# ============================================================
# EDGE BUILDER
# ============================================================

def build_edge(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    source_map: dict,
    target_map: dict,
    time: pd.Series,
):
    """
    Return (edge_index, edge_time) for one relation, dropping any row whose
    endpoints are not present in the corresponding node index.
    """

    source_ids = frame[source_column].astype(str).map(source_map)
    target_ids = frame[target_column].astype(str).map(target_map)

    valid = source_ids.notna() & target_ids.notna()

    if not valid.any():
        return None, None

    edge_index = torch.tensor(
        np.stack(
            [
                source_ids[valid].to_numpy(dtype=np.int64),
                target_ids[valid].to_numpy(dtype=np.int64),
            ]
        ),
        dtype=torch.long,
    )

    edge_time = torch.tensor(
        time.loc[frame.index[valid]].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    return edge_index, edge_time


def attach(data: HeteroData, relation: tuple, edge_index, edge_time) -> None:

    if edge_index is None or edge_index.shape[1] == 0:
        return

    data[relation].edge_index = edge_index
    data[relation].edge_time = edge_time


# ============================================================
# CONSTRUCTION (shared by the batch script and live scoring)
# ============================================================

def build_hetero_graph(
    df: pd.DataFrame,
    scalers: dict = None,
    with_labels: bool = True,
):
    """
    Build the heterogeneous graph from an event frame.

    Args:
        df: events, with `timestamp` already parsed to datetime.
        scalers: feature statistics per node type. Pass the stored training
            scalers when serving; pass None (or an empty dict) to fit them,
            which is what the training run does.
        with_labels: attach `y` and the train/val/test masks. Live traffic
            has no labels, so the streaming path sets this False.

    Returns:
        (data, scalers, accounts)
    """

    if scalers is None:
        scalers = {}

    time, span = normalised_time(
        df,
        scalers.get(TIME_SPAN_KEY),
    )

    if TIME_SPAN_KEY not in scalers and span is not None:
        scalers[TIME_SPAN_KEY] = span

    event_type = df["event_type"].astype(str).str.upper()

    # --------------------------------------------------------
    # Node indexes
    # --------------------------------------------------------

    accounts, account_map = build_account_index(df)

    device_map = build_entity_index(
        df["device_id"] if "device_id" in df.columns else pd.Series(dtype=str)
    )

    vpa_values = pd.concat(
        [
            df.get("source_vpa", pd.Series(dtype=str)),
            df.get("target_vpa", pd.Series(dtype=str)),
        ]
    )

    vpa_map = build_entity_index(vpa_values)

    merchant_map = build_entity_index(
        df["merchant_id"]
        if "merchant_id" in df.columns
        else pd.Series(dtype=str)
    )

    data = HeteroData()

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    data["account"].x = scale(
        build_features(df, accounts),
        "account",
        scalers,
    )

    entity_columns = {
        "device": "device_id",
        "vpa": "source_vpa",
        "merchant": "merchant_id",
    }

    entity_maps = {
        "device": device_map,
        "vpa": vpa_map,
        "merchant": merchant_map,
    }

    for node_type in ENTITY_TYPES:

        index_map = entity_maps[node_type]

        if not index_map:
            continue

        data[node_type].x = scale(
            entity_features(
                df,
                entity_columns[node_type],
                index_map,
                time,
            ),
            node_type,
            scalers,
        )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    transfers = df[event_type == "TRANSFER"]

    edge_index, edge_time = build_edge(
        transfers,
        "source_account",
        "target_account",
        account_map,
        account_map,
        time,
    )

    attach(
        data,
        ("account", "transfers", "account"),
        edge_index,
        edge_time,
    )

    # Reverse transfers, so money received also informs an account.
    if edge_index is not None:
        attach(
            data,
            ("account", "rev_transfers", "account"),
            edge_index.flip(0),
            edge_time,
        )

    beneficiary = df[event_type == "BENEFICIARY_ADD"]

    attach(
        data,
        ("account", "adds_beneficiary", "account"),
        *build_edge(
            beneficiary,
            "source_account",
            "target_account",
            account_map,
            account_map,
            time,
        ),
    )

    paired = [
        ("device", "device_id", "logs_in", "rev_logs_in", df),
        ("vpa", "source_vpa", "owns", "rev_owns", df),
        (
            "merchant",
            "merchant_id",
            "collects_at",
            "rev_collects_at",
            df[event_type == "MERCHANT_COLLECTION"],
        ),
    ]

    for node_type, column, forward, reverse, frame in paired:

        index_map = entity_maps[node_type]

        if not index_map:
            continue

        edge_index, edge_time = build_edge(
            frame,
            "source_account",
            column,
            account_map,
            index_map,
            time,
        )

        attach(
            data,
            ("account", forward, node_type),
            edge_index,
            edge_time,
        )

        if edge_index is not None:
            attach(
                data,
                (node_type, reverse, "account"),
                edge_index.flip(0),
                edge_time,
            )

    # --------------------------------------------------------
    # Labels and splits
    # --------------------------------------------------------

    if with_labels:

        labels = build_labels(df, account_map)

        train_mask, val_mask, test_mask = build_masks(labels)

        data["account"].y = labels
        data["account"].train_mask = train_mask
        data["account"].val_mask = val_mask
        data["account"].test_mask = test_mask

    data.account_ids = accounts

    return data, scalers, accounts


# ============================================================
# MAIN
# ============================================================

def prepare_hetero_graph() -> HeteroData:

    print("=" * 60)
    print("HETEROGENEOUS GRAPH PREPARATION")
    print("=" * 60)

    df = load_events()

    print(f"Events loaded: {len(df):,}")

    data, scalers, accounts = build_hetero_graph(df)

    print(
        f"Nodes - "
        + ", ".join(
            f"{node_type}: {data[node_type].x.shape[0]}"
            for node_type in data.node_types
        )
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, OUTPUT_FILE)

    # The scaler is what makes inductive serving possible: live features must
    # be standardised with these statistics, not with statistics refitted on
    # whatever traffic is in memory.
    SCALER_FILE.write_text(
        json.dumps(scalers, indent=2),
        encoding="utf-8",
    )

    labels = data["account"].y

    print()
    print(data)

    print()

    for relation in data.edge_types:
        print(
            f"{'__'.join(relation):40s} "
            f"{data[relation].edge_index.shape[1]:>6,} edges"
        )

    print()
    print(f"Fraud-labelled accounts: {int(labels.sum())}")

    print(
        "Split (train/val/test): "
        f"{int(data['account'].train_mask.sum())} / "
        f"{int(data['account'].val_mask.sum())} / "
        f"{int(data['account'].test_mask.sum())}"
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SCALER_FILE}")
    print("=" * 60)

    return data


if __name__ == "__main__":
    prepare_hetero_graph()
