"""Compatibility wrapper that exposes the `predict_model` function from the
`cgcnn` package. The CLI is removed and can be used from Python REPL or scripts
by importing `cgcnn.inference.predict_model`.
"""

from cgcnn.inference import predict_model

__all__ = ["predict_model"]
