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

Output:
    data/processed/hetero_graph_data.pt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from graph.prepare_graph import (
    build_account_index,
    build_features,
    build_labels,
    build_masks,
    load_events,
    standardise,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"


# ============================================================
# TIME NORMALISATION
# ============================================================

def normalised_time(df: pd.DataFrame) -> pd.Series:
    """
    Map timestamps to [0, 1] across the dataset span.
    0 = first event in the dataset, 1 = most recent event.
    """

    seconds = (
        df["timestamp"].astype("int64") / 1_000_000_000.0
    )

    seconds = seconds.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    start = seconds.min()
    end = seconds.max()

    span = end - start

    if not np.isfinite(span) or span <= 0:
        return pd.Series(
            np.zeros(len(df)),
            index=df.index,
            dtype=float,
        )

    return ((seconds - start) / span).fillna(0.0)


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
) -> torch.Tensor:
    """
    Generic activity profile for a non-account node type:

        [ log(event count),
          log(distinct accounts),
          log1p(mean amount),
          mean normalised event time ]
    """

    if not index_map:
        return torch.zeros((0, 4), dtype=torch.float)

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

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return torch.tensor(
        standardise(features),
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
# MAIN
# ============================================================

def prepare_hetero_graph() -> HeteroData:

    print("=" * 60)
    print("HETEROGENEOUS GRAPH PREPARATION")
    print("=" * 60)

    df = load_events()

    print(f"Events loaded: {len(df):,}")

    time = normalised_time(df)

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

    print(
        f"Nodes - account: {len(account_map)}, "
        f"device: {len(device_map)}, "
        f"vpa: {len(vpa_map)}, "
        f"merchant: {len(merchant_map)}"
    )

    data = HeteroData()

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    data["account"].x = torch.tensor(
        standardise(build_features(df, accounts)),
        dtype=torch.float,
    )

    if device_map:
        data["device"].x = entity_features(
            df, "device_id", device_map, time
        )

    if vpa_map:
        data["vpa"].x = entity_features(
            df, "source_vpa", vpa_map, time
        )

    if merchant_map:
        data["merchant"].x = entity_features(
            df, "merchant_id", merchant_map, time
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

    if device_map:

        edge_index, edge_time = build_edge(
            df,
            "source_account",
            "device_id",
            account_map,
            device_map,
            time,
        )

        attach(
            data,
            ("account", "logs_in", "device"),
            edge_index,
            edge_time,
        )

        if edge_index is not None:
            attach(
                data,
                ("device", "rev_logs_in", "account"),
                edge_index.flip(0),
                edge_time,
            )

    if vpa_map:

        edge_index, edge_time = build_edge(
            df,
            "source_account",
            "source_vpa",
            account_map,
            vpa_map,
            time,
        )

        attach(
            data,
            ("account", "owns", "vpa"),
            edge_index,
            edge_time,
        )

        if edge_index is not None:
            attach(
                data,
                ("vpa", "rev_owns", "account"),
                edge_index.flip(0),
                edge_time,
            )

    if merchant_map:

        collections = df[event_type == "MERCHANT_COLLECTION"]

        edge_index, edge_time = build_edge(
            collections,
            "source_account",
            "merchant_id",
            account_map,
            merchant_map,
            time,
        )

        attach(
            data,
            ("account", "collects_at", "merchant"),
            edge_index,
            edge_time,
        )

        if edge_index is not None:
            attach(
                data,
                ("merchant", "rev_collects_at", "account"),
                edge_index.flip(0),
                edge_time,
            )

    # --------------------------------------------------------
    # Labels and splits
    # --------------------------------------------------------

    labels = build_labels(df, account_map)

    train_mask, val_mask, test_mask = build_masks(labels)

    data["account"].y = labels
    data["account"].train_mask = train_mask
    data["account"].val_mask = val_mask
    data["account"].test_mask = test_mask

    data.account_ids = accounts

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, OUTPUT_FILE)

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
        f"{int(train_mask.sum())} / "
        f"{int(val_mask.sum())} / "
        f"{int(test_mask.sum())}"
    )

    print(
        "Fraud per split: "
        f"{int(labels[train_mask].sum())} / "
        f"{int(labels[val_mask].sum())} / "
        f"{int(labels[test_mask].sum())}"
    )

    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 60)

    return data


if __name__ == "__main__":
    prepare_hetero_graph()
