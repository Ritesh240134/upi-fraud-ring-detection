"""
DEPRECATED - kept only so existing commands keep working.

This module used to define a second, separate FastAPI app whose /score
endpoint read `data/processed/account_labels.csv` and returned 0.95 for any
account labelled fraudulent. That is the ground-truth label, so it reported
near-perfect risk on known accounts and told you nothing about new ones.

`backend.main` now serves real model scores from the trained temporal
heterogeneous GNN and supersedes this module entirely.

    uvicorn backend.main:app --reload

The name `app` below is re-exported so the old command still starts the
current API rather than the label-reading one.
"""

import warnings

from backend.main import app  # noqa: F401  (re-exported on purpose)


warnings.warn(
    "backend.realtime_scorer is deprecated; use backend.main:app instead. "
    "Its /score endpoint returned ground-truth labels rather than model "
    "predictions.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["app"]
