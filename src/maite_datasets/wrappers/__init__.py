import importlib.util

__all__ = []

if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None:
    from ._torch import TorchvisionWrapper

    __all__ += ["TorchvisionWrapper"]
