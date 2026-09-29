"""Compatibility imports for :mod:`cambam_builder.native.region`."""

from .native.region import *
from .native import region as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
