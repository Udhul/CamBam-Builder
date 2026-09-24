"""Compatibility imports for :mod:`cambam_builder.native.project`."""

from .native.project import *
from .native import project as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
