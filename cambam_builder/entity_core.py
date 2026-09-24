"""Compatibility imports for :mod:`cambam_builder.native.core`."""

from .native.core import *
from .native import core as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
