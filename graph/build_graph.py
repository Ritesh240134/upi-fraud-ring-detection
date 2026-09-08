"""
SUPERSEDED - kept for reference, not part of the active pipeline.

`graph/prepare_hetero_graph.py` replaces this module. It builds the same node
types plus per-relation `edge_time`, real (non-constant) features for device /
vpa / merchant nodes, and stratified train/val/test masks. Nothing imports
this file, and its output `data/processed/upi_graph.pt` is no longer read by
any script.

--------------------------------------------------------------------------

Build a heterogeneous UPI graph from the synthetic event stream.

Node types:
    account
    device
    vpa
    merchant

Relations:
    account -> transfers -> account
    account -> logs_in -> device
    account -> adds_beneficiary -> account
    account -> collects_at -> merchant
    account -> owns -> vpa
"""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/synthetic/upi_events.csv")
OUTPUT_FILE = Path("data/processed/upi_graph.pt")


# ============================================================
# LOAD DATA
# ============================================================

def load_events() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================
# CREATE NODE MAPPINGS
# ============================================================

def create_node_mappings(df: pd.DataFrame):

    accounts = sorted(
        set(df["source_account"].dropna())
        | set(df["target_account"].dropna())
    )

    devices = sorted(
        set(df["device_id"].dropna())
    )

    vpas = sorted(
        set(df["source_vpa"].dropna())
        | set(df["target_vpa"].dropna())
    )

    merchants = sorted(
        set(df["merchant_id"].dropna())
    )

    # Remove empty values
    accounts = [
        x for x in accounts
        if str(x).strip()
    ]

    devices = [
        x for x in devices
        if str(x).strip()
    ]

    vpas = [
        x for x in vpas
        if str(x).strip()
    ]

    merchants = [
        x for x in merchants
        if str(x).strip()
    ]

    mappings = {
        "account": {
            value: index
            for index, value in enumerate(accounts)
        },
        "device": {
            value: index
            for index, value in enumerate(devices)
        },
        "vpa": {
            value: index
            for index, value in enumerate(vpas)
        },
        "merchant": {
            value: index
            for index, value in enumerate(merchants)
        },
    }

    return mappings


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(df: pd.DataFrame) -> HeteroData:

    mappings = create_node_mappings(df)

    data = HeteroData()

    # --------------------------------------------------------
    # Node counts
    # --------------------------------------------------------

    data["account"].num_nodes = len(
        mappings["account"]
    )

    data["device"].num_nodes = len(
        mappings["device"]
    )

    data["vpa"].num_nodes = len(
        mappings["vpa"]
    )

    data["merchant"].num_nodes = len(
        mappings["merchant"]
    )

    # --------------------------------------------------------
    # Account -> Account : TRANSFER
    # --------------------------------------------------------

    transfer_edges = []

    transfer_amounts = []

    transfer_times = []

    transfer_labels = []

    for _, row in df[
        df["event_type"] == "TRANSFER"
    ].iterrows():

        source = row["source_account"]
        target = row["target_account"]

        if (
            source not in mappings["account"]
            or target not in mappings["account"]
        ):
            continue

        transfer_edges.append(
            [
                mappings["account"][source],
                mappings["account"][target],
            ]
        )

        transfer_amounts.append(
            float(row["amount"])
        )

        transfer_times.append(
            row["timestamp"].timestamp()
        )

        transfer_labels.append(
            int(row["is_fraud"])
        )

    if transfer_edges:

        data[
            "account",
            "transfers",
            "account",
        ].edge_index = torch.tensor(
            transfer_edges,
            dtype=torch.long,
        ).t().contiguous()

        data[
            "account",
            "transfers",
            "account",
        ].edge_attr = torch.tensor(
            transfer_amounts,
            dtype=torch.float32,
        ).view(-1, 1)

        data[
            "account",
            "transfers",
            "account",
        ].edge_time = torch.tensor(
            transfer_times,
            dtype=torch.float64,
        )

        data[
            "account",
            "transfers",
            "account",
        ].edge_label = torch.tensor(
            transfer_labels,
            dtype=torch.long,
        )

    # --------------------------------------------------------
    # Account -> Device : LOGIN
    # --------------------------------------------------------

    login_edges = []

    login_times = []

    for _, row in df[
        df["event_type"] == "LOGIN"
    ].iterrows():

        account = row["source_account"]
        device = row["device_id"]

        if (
            account not in mappings["account"]
            or device not in mappings["device"]
        ):
            continue

        login_edges.append(
            [
                mappings["account"][account],
                mappings["device"][device],
            ]
        )

        login_times.append(
            row["timestamp"].timestamp()
        )

    if login_edges:

        data[
            "account",
            "logs_in",
            "device",
        ].edge_index = torch.tensor(
            login_edges,
            dtype=torch.long,
        ).t().contiguous()

        data[
            "account",
            "logs_in",
            "device",
        ].edge_time = torch.tensor(
            login_times,
            dtype=torch.float64,
        )

    # --------------------------------------------------------
    # Account -> Account : BENEFICIARY_ADD
    # --------------------------------------------------------

    beneficiary_edges = []

    beneficiary_times = []

    for _, row in df[
        df["event_type"] == "BENEFICIARY_ADD"
    ].iterrows():

        source = row["source_account"]
        target = row["target_account"]

        if (
            source not in mappings["account"]
            or target not in mappings["account"]
        ):
            continue

        beneficiary_edges.append(
            [
                mappings["account"][source],
                mappings["account"][target],
            ]
        )

        beneficiary_times.append(
            row["timestamp"].timestamp()
        )

    if beneficiary_edges:

        data[
            "account",
            "adds_beneficiary",
            "account",
        ].edge_index = torch.tensor(
            beneficiary_edges,
            dtype=torch.long,
        ).t().contiguous()

        data[
            "account",
            "adds_beneficiary",
            "account",
        ].edge_time = torch.tensor(
            beneficiary_times,
            dtype=torch.float64,
        )

    # --------------------------------------------------------
    # Account -> Merchant : COLLECTION
    # --------------------------------------------------------

    merchant_edges = []

    merchant_amounts = []

    merchant_times = []

    merchant_labels = []

    for _, row in df[
        df["event_type"] == "MERCHANT_COLLECTION"
    ].iterrows():

        account = row["source_account"]
        merchant = row["merchant_id"]

        if (
            account not in mappings["account"]
            or merchant not in mappings["merchant"]
        ):
            continue

        merchant_edges.append(
            [
                mappings["account"][account],
                mappings["merchant"][merchant],
            ]
        )

        merchant_amounts.append(
            float(row["amount"])
        )

        merchant_times.append(
            row["timestamp"].timestamp()
        )

        merchant_labels.append(
            int(row["is_fraud"])
        )

    if merchant_edges:

        data[
            "account",
            "collects_at",
            "merchant",
        ].edge_index = torch.tensor(
            merchant_edges,
            dtype=torch.long,
        ).t().contiguous()

        data[
            "account",
            "collects_at",
            "merchant",
        ].edge_attr = torch.tensor(
            merchant_amounts,
            dtype=torch.float32,
        ).view(-1, 1)

        data[
            "account",
            "collects_at",
            "merchant",
        ].edge_time = torch.tensor(
            merchant_times,
            dtype=torch.float64,
        )

        data[
            "account",
            "collects_at",
            "merchant",
        ].edge_label = torch.tensor(
            merchant_labels,
            dtype=torch.long,
        )

    # --------------------------------------------------------
    # Account -> VPA : OWNS
    # --------------------------------------------------------

    account_vpa_edges = []

    account_to_vpa = (
        df[
            [
                "source_account",
                "source_vpa",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    for _, row in account_to_vpa.iterrows():

        account = row["source_account"]
        vpa = row["source_vpa"]

        if (
            account in mappings["account"]
            and vpa in mappings["vpa"]
        ):

            account_vpa_edges.append(
                [
                    mappings["account"][account],
                    mappings["vpa"][vpa],
                ]
            )

    if account_vpa_edges:

        data[
            "account",
            "owns",
            "vpa",
        ].edge_index = torch.tensor(
            account_vpa_edges,
            dtype=torch.long,
        ).t().contiguous()

    return data


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(data: HeteroData) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        data,
        OUTPUT_FILE,
    )

    print()
    print("=" * 60)
    print("Heterogeneous UPI graph created")
    print("=" * 60)

    print("\nNode types:")
    for node_type in data.node_types:
        print(
            f"  {node_type}: "
            f"{data[node_type].num_nodes}"
        )

    print("\nEdge types:")
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index

        print(
            f"  {edge_type}: "
            f"{edge_index.shape[1]} edges"
        )

    print()
    print(f"Saved graph: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading synthetic UPI events...")

    df = load_events()

    print(
        f"Loaded {len(df)} events."
    )

    print("Building heterogeneous graph...")

    graph = build_graph(df)

    save_graph(graph)


if __name__ == "__main__":
    main()