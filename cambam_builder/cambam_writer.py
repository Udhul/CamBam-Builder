"""Compatibility imports for :mod:`cambam_builder.native.writer`."""

from .native.writer import *
from .native import writer as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
