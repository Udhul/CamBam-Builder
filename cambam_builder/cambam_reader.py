"""Compatibility imports for :mod:`cambam_builder.native.reader`."""

from .native.reader import *
from .native import reader as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
