"""
UPI Fraud Ring Detection
Synthetic UPI-like streaming data simulator.

Generates:
- Accounts
- Devices
- VPAs
- Merchants
- Transfers
- Logins
- Beneficiary additions
- Merchant collections

All data is synthetic.
"""

from __future__ import annotations

import csv
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_ACCOUNTS = 500
NUM_DEVICES = 120
NUM_VPAS = 500
NUM_MERCHANTS = 80
NUM_EVENTS = 5000

START_TIME = datetime(2026, 1, 1, 0, 0, 0)

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_FILE = OUTPUT_DIR / "upi_events.csv"


# ============================================================
# EVENT STRUCTURE
# ============================================================

@dataclass
class UPIEvent:
    event_id: str
    timestamp: str
    event_type: str

    source_account: str
    target_account: str

    device_id: str
    source_vpa: str
    target_vpa: str
    merchant_id: str

    amount: float

    is_fraud: int
    typology: str
    ring_id: str


# ============================================================
# SIMULATOR
# ============================================================

class UPISimulator:

    def __init__(
        self,
        seed: int = SEED,
        num_accounts: int = NUM_ACCOUNTS,
        num_devices: int = NUM_DEVICES,
        num_vpas: int = NUM_VPAS,
        num_merchants: int = NUM_MERCHANTS,
    ):
        self.random = random.Random(seed)

        # Event ids are drawn from their own stream so that generating
        # them cannot shift the sequence of accounts, amounts and times
        # produced by `self.random`.
        self.id_random = random.Random(seed + 1)

        self.num_accounts = num_accounts
        self.num_devices = num_devices
        self.num_vpas = num_vpas
        self.num_merchants = num_merchants

        self.accounts = [
            f"ACC_{i:05d}"
            for i in range(num_accounts)
        ]

        self.devices = [
            f"DEV_{i:04d}"
            for i in range(num_devices)
        ]

        self.vpas = [
            f"VPA_{i:05d}"
            for i in range(num_vpas)
        ]

        self.merchants = [
            f"MER_{i:04d}"
            for i in range(num_merchants)
        ]

        # One VPA per account in the basic simulator.
        self.account_to_vpa = {
            account: self.vpas[i]
            for i, account in enumerate(self.accounts)
        }

        self.events: list[UPIEvent] = []

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------


    def new_event_id(self) -> str:
        """
        Deterministic RFC-4122 version-4 identifier drawn from the seeded RNG.

        `uuid.uuid4()` reads the OS entropy pool and ignores `seed`, so the
        `event_id` column used to differ on every run even though the events
        themselves were identical. Drawing the bits from the seeded RNG makes
        the whole file byte-for-byte reproducible.
        """

        return str(
            uuid.UUID(int=self.id_random.getrandbits(128), version=4)
        )

    def random_account(self) -> str:
        return self.random.choice(self.accounts)

    def random_device(self) -> str:
        return self.random.choice(self.devices)

    def random_merchant(self) -> str:
        return self.random.choice(self.merchants)

    def random_amount(self) -> float:
        return round(
            self.random.uniform(20.0, 25000.0),
            2,
        )

    def random_timestamp(self, index: int) -> str:
        """
        Generate monotonically increasing event timestamps.
        """
        seconds = self.random.randint(1, 30)

        if index == 0:
            current_time = START_TIME
        else:
            previous = datetime.fromisoformat(
                self.events[-1].timestamp
            )
            current_time = previous + timedelta(
                seconds=seconds
            )

        return current_time.isoformat()

    # --------------------------------------------------------
    # Event creators
    # --------------------------------------------------------

    def create_transfer(
        self,
        timestamp: str,
        source: str,
        target: str,
        device: str,
        amount: float,
        is_fraud: int = 0,
        typology: str = "NORMAL",
        ring_id: str = "",
    ) -> UPIEvent:

        return UPIEvent(
            event_id=self.new_event_id(),
            timestamp=timestamp,
            event_type="TRANSFER",
            source_account=source,
            target_account=target,
            device_id=device,
            source_vpa=self.account_to_vpa[source],
            target_vpa=self.account_to_vpa[target],
            merchant_id="",
            amount=amount,
            is_fraud=is_fraud,
            typology=typology,
            ring_id=ring_id,
        )

    def create_login(
        self,
        timestamp: str,
        account: str,
        device: str,
    ) -> UPIEvent:

        return UPIEvent(
            event_id=self.new_event_id(),
            timestamp=timestamp,
            event_type="LOGIN",
            source_account=account,
            target_account="",
            device_id=device,
            source_vpa=self.account_to_vpa[account],
            target_vpa="",
            merchant_id="",
            amount=0.0,
            is_fraud=0,
            typology="NORMAL",
            ring_id="",
        )

    def create_beneficiary(
        self,
        timestamp: str,
        source: str,
        target: str,
        device: str,
    ) -> UPIEvent:

        return UPIEvent(
            event_id=self.new_event_id(),
            timestamp=timestamp,
            event_type="BENEFICIARY_ADD",
            source_account=source,
            target_account=target,
            device_id=device,
            source_vpa=self.account_to_vpa[source],
            target_vpa=self.account_to_vpa[target],
            merchant_id="",
            amount=0.0,
            is_fraud=0,
            typology="NORMAL",
            ring_id="",
        )

    def create_merchant_collection(
        self,
        timestamp: str,
        account: str,
        merchant: str,
        device: str,
        amount: float,
        is_fraud: int = 0,
        typology: str = "NORMAL",
        ring_id: str = "",
    ) -> UPIEvent:

        return UPIEvent(
            event_id=self.new_event_id(),
            timestamp=timestamp,
            event_type="MERCHANT_COLLECTION",
            source_account=account,
            target_account="",
            device_id=device,
            source_vpa=self.account_to_vpa[account],
            target_vpa="",
            merchant_id=merchant,
            amount=amount,
            is_fraud=is_fraud,
            typology=typology,
            ring_id=ring_id,
        )

    # --------------------------------------------------------
    # Normal traffic
    # --------------------------------------------------------

    def generate_normal_events(self, count: int) -> None:

        for i in range(count):

            timestamp = self.random_timestamp(i)

            event_type = self.random.choices(
                [
                    "TRANSFER",
                    "LOGIN",
                    "BENEFICIARY_ADD",
                    "MERCHANT_COLLECTION",
                ],
                weights=[65, 15, 10, 10],
                k=1,
            )[0]

            source = self.random_account()
            device = self.random_device()

            if event_type == "TRANSFER":

                target = self.random_account()

                while target == source:
                    target = self.random_account()

                event = self.create_transfer(
                    timestamp=timestamp,
                    source=source,
                    target=target,
                    device=device,
                    amount=self.random_amount(),
                )

            elif event_type == "LOGIN":

                event = self.create_login(
                    timestamp=timestamp,
                    account=source,
                    device=device,
                )

            elif event_type == "BENEFICIARY_ADD":

                target = self.random_account()

                while target == source:
                    target = self.random_account()

                event = self.create_beneficiary(
                    timestamp=timestamp,
                    source=source,
                    target=target,
                    device=device,
                )

            else:

                event = self.create_merchant_collection(
                    timestamp=timestamp,
                    account=source,
                    merchant=self.random_merchant(),
                    device=device,
                    amount=self.random_amount(),
                )

            self.events.append(event)

    # --------------------------------------------------------
    # T1 - Fan-out layering
    # --------------------------------------------------------

    def generate_t1_fanout(self) -> None:

        ring_id = "RING_T1_001"

        victim = self.random_account()

        mules = self.random.sample(
            [a for a in self.accounts if a != victim],
            10,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        for i, mule in enumerate(mules):

            timestamp = (
                base_time + timedelta(minutes=i)
            ).isoformat()

            event = self.create_transfer(
                timestamp=timestamp,
                source=victim,
                target=mule,
                device=self.random_device(),
                amount=round(
                    self.random.uniform(5000, 15000),
                    2,
                ),
                is_fraud=1,
                typology="T1_FAN_OUT_LAYERING",
                ring_id=ring_id,
            )

            self.events.append(event)

    # --------------------------------------------------------
    # T2 - Smurfing
    # --------------------------------------------------------

    def generate_t2_smurfing(self) -> None:

        ring_id = "RING_T2_001"

        victim = self.random_account()

        mules = self.random.sample(
            [a for a in self.accounts if a != victim],
            8,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        for i in range(20):

            mule = mules[i % len(mules)]

            timestamp = (
                base_time + timedelta(seconds=i * 30)
            ).isoformat()

            event = self.create_transfer(
                timestamp=timestamp,
                source=victim,
                target=mule,
                device=self.random_device(),
                amount=round(
                    self.random.uniform(100, 499),
                    2,
                ),
                is_fraud=1,
                typology="T2_SMURFING",
                ring_id=ring_id,
            )

            self.events.append(event)

    # --------------------------------------------------------
    # T3 - Mule farm burst
    # --------------------------------------------------------

    def generate_t3_mule_farm(self) -> None:

        ring_id = "RING_T3_001"

        mules = self.random.sample(
            self.accounts,
            20,
        )

        devices = self.random.sample(
            self.devices,
            3,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        for i, account in enumerate(mules):

            timestamp = (
                base_time + timedelta(seconds=i * 20)
            ).isoformat()

            event = self.create_login(
                timestamp=timestamp,
                account=account,
                device=devices[i % len(devices)],
            )

            event.is_fraud = 1
            event.typology = "T3_MULE_FARM_BURST"
            event.ring_id = ring_id

            self.events.append(event)

    # --------------------------------------------------------
    # T4 - Circular round trip
    # --------------------------------------------------------

    def generate_t4_cycle(self) -> None:

        ring_id = "RING_T4_001"

        accounts = self.random.sample(
            self.accounts,
            6,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        for i in range(len(accounts)):

            source = accounts[i]
            target = accounts[
                (i + 1) % len(accounts)
            ]

            timestamp = (
                base_time + timedelta(minutes=i)
            ).isoformat()

            event = self.create_transfer(
                timestamp=timestamp,
                source=source,
                target=target,
                device=self.random_device(),
                amount=7500.00,
                is_fraud=1,
                typology="T4_CIRCULAR_ROUND_TRIP",
                ring_id=ring_id,
            )

            self.events.append(event)

    # --------------------------------------------------------
    # T5 - Merchant collusion
    # --------------------------------------------------------

    def generate_t5_merchant_collusion(self) -> None:

        ring_id = "RING_T5_001"

        merchant = self.random_merchant()

        accounts = self.random.sample(
            self.accounts,
            10,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        for i, account in enumerate(accounts):

            timestamp = (
                base_time + timedelta(minutes=i)
            ).isoformat()

            event = self.create_merchant_collection(
                timestamp=timestamp,
                account=account,
                merchant=merchant,
                device=self.random_device(),
                amount=round(
                    self.random.uniform(3000, 12000),
                    2,
                ),
                is_fraud=1,
                typology="T5_MERCHANT_COLLUSION",
                ring_id=ring_id,
            )

            self.events.append(event)

    # --------------------------------------------------------
    # T6 - Social engineering harvest
    # --------------------------------------------------------

    def generate_t6_social_engineering(self) -> None:

        ring_id = "RING_T6_001"

        victims = self.random.sample(
            self.accounts,
            12,
        )

        aggregation_account = self.random.choice(
            [
                a for a in self.accounts
                if a not in victims
            ]
        )

        dispersal_accounts = self.random.sample(
            [
                a for a in self.accounts
                if a not in victims
                and a != aggregation_account
            ],
            8,
        )

        base_time = datetime.fromisoformat(
            self.events[-1].timestamp
        )

        # Victims -> aggregation account
        for i, victim in enumerate(victims):

            timestamp = (
                base_time + timedelta(seconds=i * 20)
            ).isoformat()

            event = self.create_transfer(
                timestamp=timestamp,
                source=victim,
                target=aggregation_account,
                device=self.random_device(),
                amount=round(
                    self.random.uniform(500, 3000),
                    2,
                ),
                is_fraud=1,
                typology="T6_SOCIAL_ENGINEERING_HARVEST",
                ring_id=ring_id,
            )

            self.events.append(event)

        # Aggregation -> dispersal layer
        for i, target in enumerate(dispersal_accounts):

            timestamp = (
                base_time
                + timedelta(minutes=5, seconds=i * 20)
            ).isoformat()

            event = self.create_transfer(
                timestamp=timestamp,
                source=aggregation_account,
                target=target,
                device=self.random_device(),
                amount=round(
                    self.random.uniform(1000, 5000),
                    2,
                ),
                is_fraud=1,
                typology="T6_SOCIAL_ENGINEERING_HARVEST",
                ring_id=ring_id,
            )

            self.events.append(event)

    # --------------------------------------------------------
    # Run simulator
    # --------------------------------------------------------

    def run(self) -> list[UPIEvent]:

        print("Generating normal UPI-like events...")
        self.generate_normal_events(NUM_EVENTS)

        print("Generating T1 fraud ring...")
        self.generate_t1_fanout()

        print("Generating T2 fraud ring...")
        self.generate_t2_smurfing()

        print("Generating T3 fraud ring...")
        self.generate_t3_mule_farm()

        print("Generating T4 fraud ring...")
        self.generate_t4_cycle()

        print("Generating T5 fraud ring...")
        self.generate_t5_merchant_collusion()

        print("Generating T6 fraud ring...")
        self.generate_t6_social_engineering()

        self.events.sort(
            key=lambda event: event.timestamp
        )

        return self.events

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    def save(self) -> None:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.events:
            raise RuntimeError(
                "No events generated."
            )

        with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=asdict(
                    self.events[0]
                ).keys(),
            )

            writer.writeheader()

            for event in self.events:
                writer.writerow(
                    asdict(event)
                )

        print()
        print("=" * 60)
        print("UPI-Sim generation completed")
        print("=" * 60)
        print(f"Total events : {len(self.events)}")
        print(f"Output file  : {OUTPUT_FILE}")
        print()

        fraud_count = sum(
            event.is_fraud
            for event in self.events
        )

        print(f"Fraud events : {fraud_count}")
        print(
            f"Normal events: "
            f"{len(self.events) - fraud_count}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    simulator = UPISimulator()

    simulator.run()

    simulator.save()