"""Backwards-compatibility wrappers for the cgcnn package.

This file no longer provides CLI. Import the training routines via:

    from cgcnn.training import train_model

and call `train_model(...)` from the Python REPL or your Python script.
"""

from cgcnn.training import train_model  # re-export

__all__ = ["train_model"]
