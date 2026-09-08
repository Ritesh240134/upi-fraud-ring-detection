"""
Prepare the homogeneous account transfer graph for the GraphSAGE baseline.

Node type : account
Edge type : account -> account (a transfer occurred)

IMPORTANT
---------
Node features are derived from transaction *behaviour* only.
Nothing derived from `is_fraud`, `ring_id` or `typology` is ever used as a
feature - those columns are the label and using them would leak it.

Output:
    data/processed/graph_data.pt
        {
            "data"           : torch_geometric.data.Data
            "accounts"       : list[str]
            "account_to_id"  : dict[str, int]
            "feature_names"  : list[str]
        }
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "graph_data.pt"

SEED = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


FEATURE_NAMES = [
    "out_degree",
    "in_degree",
    "out_amount_log",
    "in_amount_log",
    "mean_out_amount_log",
    "mean_in_amount_log",
    "unique_out_peers",
    "unique_in_peers",
    "device_count",
    "beneficiary_adds",
    "merchant_collections",
    "night_activity_ratio",
    "median_gap_seconds_log",
]


# ============================================================
# LOAD
# ============================================================

def load_events() -> pd.DataFrame:

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Events file not found: {EVENTS_FILE}"
        )

    df = pd.read_csv(EVENTS_FILE)

    for column in ("source_account", "target_account"):
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).fillna(0.0)

    return df


# ============================================================
# ACCOUNT UNIVERSE
# ============================================================

def build_account_index(df: pd.DataFrame):

    accounts = set(
        df["source_account"].dropna().astype(str)
    )

    accounts.update(
        df["target_account"].dropna().astype(str)
    )

    accounts = sorted(
        account
        for account in accounts
        if account.strip() and account.lower() != "nan"
    )

    account_to_id = {
        account: index
        for index, account in enumerate(accounts)
    }

    return accounts, account_to_id


# ============================================================
# BEHAVIOURAL FEATURES (NO LABEL INFORMATION)
# ============================================================

def build_features(df: pd.DataFrame, accounts: list) -> np.ndarray:

    index = pd.Index(accounts, name="account")

    source = df["source_account"].astype(str)

    transfers = df[
        df["event_type"].astype(str).str.upper() == "TRANSFER"
    ].copy()

    transfer_source = transfers["source_account"].astype(str)
    transfer_target = transfers["target_account"].astype(str)

    def reindex(series: pd.Series) -> np.ndarray:
        return (
            series.reindex(index)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    # --- degrees -------------------------------------------------
    out_degree = reindex(
        transfer_source.value_counts()
    )

    in_degree = reindex(
        transfer_target.value_counts()
    )

    # --- amounts -------------------------------------------------
    out_amount = reindex(
        transfers.groupby(transfer_source)["amount"].sum()
    )

    in_amount = reindex(
        transfers.groupby(transfer_target)["amount"].sum()
    )

    mean_out_amount = reindex(
        transfers.groupby(transfer_source)["amount"].mean()
    )

    mean_in_amount = reindex(
        transfers.groupby(transfer_target)["amount"].mean()
    )

    # --- counterparty diversity ----------------------------------
    unique_out_peers = reindex(
        transfers.groupby(transfer_source)["target_account"].nunique()
    )

    unique_in_peers = reindex(
        transfers.groupby(transfer_target)["source_account"].nunique()
    )

    # --- device / other event types ------------------------------
    if "device_id" in df.columns:
        device_count = reindex(
            df.groupby(source)["device_id"].nunique()
        )
    else:
        device_count = np.zeros(len(accounts))

    event_type = df["event_type"].astype(str).str.upper()

    beneficiary_adds = reindex(
        source[event_type == "BENEFICIARY_ADD"].value_counts()
    )

    merchant_collections = reindex(
        source[event_type == "MERCHANT_COLLECTION"].value_counts()
    )

    # --- temporal behaviour --------------------------------------
    hours = df["timestamp"].dt.hour

    is_night = ((hours < 6) | (hours >= 22)).astype(float)

    night_events = reindex(
        is_night.groupby(source).sum()
    )

    total_events = reindex(
        source.value_counts()
    )

    night_activity_ratio = np.divide(
        night_events,
        total_events,
        out=np.zeros_like(night_events),
        where=total_events > 0,
    )

    ordered = df.sort_values("timestamp")

    ordered_source = ordered["source_account"].astype(str)

    gaps = (
        ordered.groupby(ordered_source)["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    median_gap = reindex(
        gaps.groupby(ordered_source).median()
    )

    # --- assemble ------------------------------------------------
    features = np.column_stack(
        [
            out_degree,
            in_degree,
            np.log1p(out_amount),
            np.log1p(in_amount),
            np.log1p(mean_out_amount),
            np.log1p(mean_in_amount),
            unique_out_peers,
            unique_in_peers,
            device_count,
            beneficiary_adds,
            merchant_collections,
            night_activity_ratio,
            np.log1p(np.clip(median_gap, 0.0, None)),
        ]
    ).astype(np.float32)

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features


def standardise(features: np.ndarray) -> np.ndarray:

    mean = features.mean(axis=0)
    std = features.std(axis=0)

    std[std < 1e-8] = 1.0

    return ((features - mean) / std).astype(np.float32)


# ============================================================
# EDGES
# ============================================================

def build_edges(df: pd.DataFrame, account_to_id: dict) -> torch.Tensor:

    transfers = df[
        df["event_type"].astype(str).str.upper() == "TRANSFER"
    ]

    source_ids = (
        transfers["source_account"].astype(str).map(account_to_id)
    )

    target_ids = (
        transfers["target_account"].astype(str).map(account_to_id)
    )

    valid = source_ids.notna() & target_ids.notna()

    edge_index = torch.tensor(
        np.stack(
            [
                source_ids[valid].to_numpy(dtype=np.int64),
                target_ids[valid].to_numpy(dtype=np.int64),
            ]
        ),
        dtype=torch.long,
    )

    return edge_index


# ============================================================
# LABELS
# ============================================================

def build_labels(df: pd.DataFrame, account_to_id: dict) -> torch.Tensor:
    """
    An account is positive when it takes part in at least one event
    flagged `is_fraud == 1`.
    """

    labels = np.zeros(len(account_to_id), dtype=np.float32)

    if "is_fraud" not in df.columns:
        return torch.tensor(labels)

    fraud = df[
        pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0) == 1
    ]

    for column in ("source_account", "target_account"):

        ids = fraud[column].astype(str).map(account_to_id).dropna()

        labels[ids.to_numpy(dtype=np.int64)] = 1.0

    return torch.tensor(labels, dtype=torch.float32)


# ============================================================
# SPLITS
# ============================================================

def build_masks(labels: torch.Tensor, seed: int = SEED):
    """
    Stratified train / validation / test split so that every split
    contains fraud accounts even though the positive rate is low.
    """

    generator = np.random.default_rng(seed)

    num_nodes = labels.shape[0]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for class_value in (0.0, 1.0):

        indices = torch.nonzero(
            labels == class_value,
            as_tuple=True,
        )[0].numpy()

        generator.shuffle(indices)

        train_end = int(round(TRAIN_FRACTION * len(indices)))

        val_end = train_end + int(
            round(VAL_FRACTION * len(indices))
        )

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask


# ============================================================
# MAIN
# ============================================================

def prepare_graph() -> Data:

    print("=" * 60)
    print("GRAPH PREPARATION (homogeneous baseline)")
    print("=" * 60)

    df = load_events()

    print(f"Events loaded: {len(df):,}")

    accounts, account_to_id = build_account_index(df)

    print(f"Account nodes: {len(accounts):,}")

    features = standardise(
        build_features(df, accounts)
    )

    edge_index = build_edges(df, account_to_id)

    labels = build_labels(df, account_to_id)

    train_mask, val_mask, test_mask = build_masks(labels)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=labels,
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "data": data,
            "accounts": accounts,
            "account_to_id": account_to_id,
            "feature_names": FEATURE_NAMES,
        },
        OUTPUT_FILE,
    )

    print(f"Node features: {tuple(data.x.shape)}")
    print(f"Feature names: {', '.join(FEATURE_NAMES)}")
    print(f"Transfer edges: {edge_index.shape[1]:,}")
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
    prepare_graph()
