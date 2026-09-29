"""Compatibility imports for :mod:`cambam_builder.native.transformations`."""

from .native.transformations import *
from .native import transformations as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
