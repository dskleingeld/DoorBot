import sys
from src.agents import Pioneer
from typing import List, Tuple, Optional
from loguru import logger
import enum
import numpy as np
from plot import Plot, report_status
from actions import Action, rot_away, rot_towards


ANGLES = np.loadtxt("angles.txt") / 180*np.pi
SIN = np.sin(ANGLES)
COS = np.cos(ANGLES)


def arg_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Anchor:
    """ finds closest point on the right side of the bot prefering
    points closer to the front"""

    def __init__(self, ranges: np.ndarray, angles: np.ndarray, prev):
        right = ranges[25:165].copy()  # dont even look to the left or back
        angles = angles[25:165]
        # slightly prefer frontal angles
        hint = abs(angles)
        hint = hint/max(hint)
        if prev is None:  # escpecially at the beginning
            right *= 1+0.5*hint
        else:
            right *= 1+0.2*hint

        # prefer a consistant choice
        if prev is not None:
            start = arg_nearest(angles, prev.angle-8)
            stop = arg_nearest(angles, prev.angle+8)
            right[start:stop] *= 0.92

        self.idx = np.argmin(right)
        self.angle = angles[self.idx]
        self.range = right[self.idx]
        self.idx += 25

    def __str__(self):
        return f"(range: {self.range:.2f}, angle: {self.angle:.2f})"


class State:
    prev_pivot: Optional[Anchor] = None
    move_to = True
    plot: Plot = Plot()
    ranges: np.ndarray = np.zeros((3, 270))


def move_along(closest: Anchor, adjust_left=False,
               adjust_right=False) -> Action:
    OPTIMAL_RANGE = 0.4
    MARGIN = 0.1
    MARGIN = 8
    target = -90
    if adjust_left:
        target += 5
    elif adjust_right:
        target -= 5

    if closest.angle > target+MARGIN:
        # pointing to far right
        print(f"move_along, rotating away: {closest}", end="\r")
        return Action.Left
    elif closest.angle < target-MARGIN:
        # pointing to far left
        print(f"move_along, rotating towards: {closest}", end="\r")
        return Action.Right
    else:
        # pointing perfectly forward
        print(f"move_along, moving forward: {closest}", end="\r")
        return Action.Forward


def move_to(closest: Anchor) -> Action:
    MARGIN = 10  # degrees
    angle = closest.angle
    if abs(angle) > MARGIN:
        print(f"move_to, rotating: {closest}", end="\r")
        return rot_towards(closest.angle)
    else:
        print(f"move_to, forward: {closest}", end="\r")
        return Action.Forward


def brain(closest: Anchor, state: State) -> Action:
    OPTIMAL_RANGE = 0.4
    MARGIN = 0.1

    if state.move_to:
        if closest.range > OPTIMAL_RANGE:
            return move_to(closest)
        else:
            state.move_to = False
            print("stopping init")
            return move_to(closest)

    if closest.range > OPTIMAL_RANGE + MARGIN:
        return move_along(closest, adjust_right=True)
    elif closest.range < OPTIMAL_RANGE - MARGIN:
        return move_along(closest, adjust_left=True)
    else:  # thus OPITMAL_RANGE - MARGIN > closest.range < OPTIMAL_RANGE:
        return move_along(closest)


def loop(agent: Pioneer, state: State):
    angles = np.loadtxt("angles.txt")
    ranges = agent.read_lidars()
    ranges = np.array(ranges)
    ranges[134] = (ranges[133] + ranges[135])/2

    pivot = Anchor(ranges, angles, state.prev_pivot)
    state.prev_pivot = pivot

    action = brain(pivot, state)
    action.perform(agent)

    x = -1*ranges*SIN
    y = ranges*COS
    state.plot.update(x, y, pivot.idx, action)
