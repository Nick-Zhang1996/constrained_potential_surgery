"""Dimensions and initial state shared by the CasADi game configurations."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseGameConfig:
    T: int = 0
    dt: float = 0
    N: int = 0
    n: int = 0
    m: int = 0
    x0: Any = None
