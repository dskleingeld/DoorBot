import numpy as np
from typing import Tuple


def measure_gap(gap, data) -> Tuple[float, float]:
    return (0, 0)


def measure_depth(gap, data) -> float:
    return 0


def scan_gaps(data: np.ndarray) -> Tuple[int, int]:
    return (0, 0)


def is_door(max: float, min: float,
            depth: float) -> Tuple[float, float]:
    return (0, 0)


def find_doors(data):
    candidates = []
    gaps = scan_gaps(data)

    for gap in gaps:
        (max, min) = measure_gap(gap, data)
        depth = measure_depth(gap, data)

        (chance, door) = is_door(max, min, depth)
        candidates.append((chance, door))

    return candidates
