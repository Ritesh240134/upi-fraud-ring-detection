"""
Create account-level fraud labels from the synthetic UPI event stream.

Label:
    0 = normal account
    1 = account involved in a fraud ring

Output:
    data/processed/account_labels.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/synthetic/upi_events.csv")
OUTPUT_FILE = Path("data/processed/account_labels.csv")


# ============================================================
# LOAD EVENTS
# ============================================================

def load_events():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# CREATE ACCOUNT LABELS
# ============================================================

def create_labels(df):

    # Collect every account appearing in the dataset
    accounts = set(
        df["source_account"].dropna().astype(str)
    )

    accounts.update(
        df["target_account"].dropna().astype(str)
    )

    # Start with every account as normal
    labels = {
        account: 0
        for account in accounts
    }

    # Find events belonging to fraud rings
    fraud_events = df[
        (df["is_fraud"] == 1)
        & (df["ring_id"].notna())
    ]

    # Mark accounts participating in fraud events
    for _, row in fraud_events.iterrows():

        source = row["source_account"]
        target = row["target_account"]

        if pd.notna(source):
            labels[str(source)] = 1

        if pd.notna(target):
            labels[str(target)] = 1

    # Convert to DataFrame
    result = pd.DataFrame(
        [
            {
                "account_id": account,
                "label": label,
            }
            for account, label in labels.items()
        ]
    )

    result = result.sort_values(
        "account_id"
    ).reset_index(drop=True)

    return result


# ============================================================
# SAVE
# ============================================================

def save_labels(result):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("ACCOUNT FRAUD LABELS CREATED")
    print("=" * 60)

    print(f"Total accounts : {len(result)}")
    print(
        f"Fraud accounts : "
        f"{(result['label'] == 1).sum()}"
    )
    print(
        f"Normal accounts: "
        f"{(result['label'] == 0).sum()}"
    )

    print()
    print("Label distribution:")
    print(result["label"].value_counts().sort_index())

    print()
    print(f"Saved labels: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading UPI events...")

    df = load_events()

    print(
        f"Loaded {len(df)} events."
    )

    print("Creating account-level fraud labels...")

    labels = create_labels(df)

    save_labels(labels)


if __name__ == "__main__":
    main()
    