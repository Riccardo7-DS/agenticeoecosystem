"""EO ecosystem utilities and lazily loaded evaluation exports."""
from .utils import *
from . import evaluation as _evaluation


def __getattr__(name: str):
    if name in _evaluation.__all__:
        value = getattr(_evaluation, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in globals() if not name.startswith("_")] + _evaluation.__all__
