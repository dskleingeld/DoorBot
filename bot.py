from src.agents import Pioneer
from typing import List, Optional
from loguru import logger
import numpy as np
from plot import Plot
from actions import Action, rot_towards, rot_towards_moving


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
    SKIP = 38

    def __init__(self, ranges: np.ndarray, angles: np.ndarray, prev):
        # dont even look to the left or back
        right = ranges[self.SKIP:165].copy()
        angles = angles[self.SKIP:165]
        # slightly prefer frontal angles
        hint = abs(angles)
        hint = hint/max(hint)

        min_d = min(right)
        if prev is None:  # escpecially at the beginning
            right *= 1+0.8*hint
        elif min_d < 0.5:  # and at close ranges
            right *= 1+0.5*hint
        else:
            right *= 1+0.5*hint

        # prefer a consistant choice
        if prev is not None:
            start = arg_nearest(angles, prev.angle-8)
            stop = arg_nearest(angles, prev.angle+8)
            right[start:stop] *= 0.90

        self.idx = np.argmin(right)
        self.angle = angles[self.idx]
        self.range = right[self.idx]
        self.idx += self.SKIP

    def __str__(self):
        return f"(range: {self.range:.2f}, angle: {self.angle:.2f})"


class State:
    prev_pivot: Optional[Anchor] = None
    move_to = True
    plot: Plot = Plot()
    ranges: np.ndarray = np.zeros((3, 270))


def move_along(closest: Anchor, adjust_left=False,
               adjust_right=False) -> Action:
    MARGIN = 5  # lower margin and the lidar location
    # changing from rotation will cause an endless wobble
    target = -92  # as 90 != 90 it seems
    if adjust_left:
        target += 10
    elif adjust_right:
        target -= 10

    rot_spd = max(abs(closest.angle - target)*1/45, 0.05)
    if closest.angle < target-MARGIN:
        # pointing to far left
        print(f"move_along, rotating towards: {target}, current: {closest}, spd: {rot_spd}",
              end="\r")
        return Action.right(rot_spd)
        # return Action.forward_right(0.4, 0.1)
    elif closest.angle > target+MARGIN:
        # pointing to far right
        print(
            f"move_along, rotating away: target: {target}, current: {closest}, spd: {rot_spd}",
            end="\r")
        return Action.left(rot_spd)
        # return rot_towards(-1*target)
    else:
        # pointing perfectly forward
        print(f"move_along, moving forward: {closest}", end="\r")
        return Action.forward(0.5)


def move_to(closest: Anchor) -> Action:
    MARGIN = 10  # degrees
    angle = closest.angle
    speed = max(closest.range*2, 0.15)
    if abs(angle) > MARGIN:
        print(f"move_to, rotating: {closest}", end="\r")
        return rot_towards(closest.angle)
    if abs(angle) > MARGIN/2:
        print(f"move_to, forward rotate: {closest}", end="\r")
        return rot_towards_moving(closest.angle, speed)
    else:
        print(f"move_to, forward: {closest}", end="\r")
        return Action.forward(speed)


def brain(closest: Anchor, state: State) -> Action:
    OPTIMAL_RANGE = 0.42
    MARGIN = 0.07

    if state.move_to:
        if closest.range > OPTIMAL_RANGE:
            return move_to(closest)
        else:
            state.move_to = False
            print("stopping init")
            return move_along(closest)

    if closest.range > OPTIMAL_RANGE + 2*MARGIN:
        state.move_to = True
        return move_to(closest)
    elif closest.range > OPTIMAL_RANGE + MARGIN/2:
        return move_along(closest, adjust_right=True)

    elif closest.range < OPTIMAL_RANGE - MARGIN/2:
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
