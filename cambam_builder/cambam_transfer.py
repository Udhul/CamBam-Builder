"""Compatibility imports for :mod:`cambam_builder.native.transfer`."""

from .native.transfer import *
from .native import transfer as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
