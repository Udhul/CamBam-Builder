"""Compatibility imports for :mod:`cambam_builder.native.cam`."""

from .native.cam import *
from .native import cam as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
