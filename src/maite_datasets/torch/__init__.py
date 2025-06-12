__all__ = []

import importlib.util

if importlib.util.find_spec("torch") is not None:
    from ._voc import VOCDetectionTorch

    __all__ += ["VOCDetectionTorch"]
