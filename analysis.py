import math
import numpy as np
from scipy import signal
from typing import Tuple, List

import far_doors as far
import close_doors as close
from doors import Door


def find_doors(data: Tuple[np.ndarray, np.ndarray],
               ranges: np.ndarray) -> List[Door]:
    doors = far.find(*data, ranges)
    doors += (close.find(*data, ranges))
    # doors = close.find(*data, ranges)

    doors.sort(key=lambda k: k.score())
    return doors


def passing_door(data: Tuple[np.ndarray, np.ndarray],
                 ranges: np.ndarray) -> bool:

    return True
