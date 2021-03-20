from typing import Tuple, List, Optional
import numpy as np
from dataclasses import dataclass
from wall_model import WallModel
from ransac import ransac
from skimage.transform import probabilistic_hough_line

MIN_LINE_SAMPLES = 10


# @dataclass
# class Line:
#     model: WallModel
#     start: Tuple[float, float]
#     end: Tuple[float, float]
#     inliers: np.ndarray  # for test only


# def find_lines(x: np.ndarray, y: np.ndarray) -> List[Line]:
#     lines = []
#     data = np.column_stack([x, y])
#     while len(data) > MIN_LINE_SAMPLES:
#         res = extract_ransac_line(data, MIN_LINE_SAMPLES)
#         if res is None:
#             break
#         line, data = res
#         lines.append(line)
#     return lines


@dataclass
class Line:
    start: Tuple[float, float]
    stop: Tuple[float, float]


def find_lines(x: np.ndarray, y: np.ndarray) -> List[Line]:
    lines = []
    def width(a): return np.abs(np.amax(a) - np.amin(a))
    def center(a): return np.amin(a) + width(a)/2
    max_width = max(width(x), width(y))
    xcent = center(x)
    ycent = center(y)

    # center x and y into a max_width by max_width grid
    x = x+(max_width/2 - center(x))
    y = y+(max_width/2 - center(y))

    # scale x and y to integer grid
    x = (x * 999/max_width).astype(int)
    y = (y * 999/max_width).astype(int)

    grid = np.zeros((1000, 1000))
    grid[y, x] = 1

    lines = probabilistic_hough_line(
        grid, line_gap=30, threshold=1, line_length=40)

    def rescale(p, cent): return p/(999/max_width) - (max_width/2 - cent)

    def rescale_line(p0, p1): return (
        (rescale(p0[0], xcent), rescale(p0[1], ycent)),
        (rescale(p1[0], xcent), rescale(p1[1], ycent)))
    lines = [rescale_line(*line) for line in lines]

    return lines


def dir_bounds(dir, origin, points) -> Tuple[float, float]:
    if dir > 0:
        end = np.amax(points)
        start = origin
    else:
        print(points.shape)
        start = np.amin(points)
        end = origin
    return start, end


def find_bounds(model, points) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    [dir, origin] = model.params
    x_bounds = dir_bounds(dir[0], origin[0], points)
    y_bounds = dir_bounds(dir[1], origin[1], points)
    return x_bounds, y_bounds


# TODO custom ransac that punishes line length
def extract_ransac_line(data, min_samples: int) -> Optional[Tuple[Line, np.ndarray]]:
    model, inliers = ransac(
        data, WallModel, min_samples=min_samples,
        residual_threshold=0.5, max_trials=1000)
    if model is None:
        return None

    # remove inliers
    updated_data = data[~inliers]

    start, end = find_bounds(model, data[inliers])
    line = Line(model, start, end, data[inliers])

    return (line, updated_data)


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    x = np.loadtxt("x.txt")
    y = np.loadtxt("y.txt")

    # lines = find_lines(x, y)
    data = np.column_stack([x, y])
    lines = find_lines(x, y)

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.2)

    for line in lines:
        line_x = np.arange(line.start[0], line.end[0])
        line_y = line.model.predict_y(line_x)
        plot_line, = ax.plot(line_x, line_y)
        color = plt.getp(plot_line, "color")
        ax.scatter(line.inliers[:, 0], line.inliers[:, 1], color=color)

    plt.show()
