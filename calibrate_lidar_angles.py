import settings
from src.env import VrepEnvironment
from src.agents import Pioneer
import numpy as np
from matplotlib import pyplot as plt
import time
import math
from scipy.signal import savgol_filter


def get_and_save_data():
    for scene, angle in zip(["0", "l90", "r90"], [(90, 180), (0, 90), (180, 270)]):
        environment = VrepEnvironment(
            settings.SCENES + '/room_calib_'+scene+'.ttt')
        # environment.start_vrep()
        environment.connect()
        environment.load_scene(settings.SCENES + '/room_calib_'+scene+'.ttt')
        environment.start_simulation()
        agent = Pioneer(environment)
        time.sleep(1)

        sample = agent.read_lidars()
        ranges = np.array(sample)[angle[0]:angle[1]]
        for _ in range(0, 29):
            sample = agent.read_lidars()
            ranges += np.array(sample)[angle[0]:angle[1]]
        ranges = ranges / 10
        np.savetxt("ranges_"+scene+".txt", ranges)

        environment.stop_simulation()
        environment.disconnect()
        del environment


# Idea do this in a loop maximizing hough transform data?
def calib(r: np.ndarray) -> np.ndarray:
    wall_normal = min(r)
    wall_normal_idx = np.argmin(r)
    angles = 180/math.pi * np.arccos(wall_normal/r)

    midpoint = 44
    angles[:wall_normal_idx] *= -1
    angles += angles[midpoint]/2
    return angles


def calib_and_save():
    r = np.loadtxt("ranges_0.txt")
    r[44] = (r[43]+r[45])/2
    a_0 = calib(r)
    a_l90 = calib(np.loadtxt("ranges_l90.txt"))
    a_r90 = calib(np.loadtxt("ranges_r90.txt"))

    angles = np.zeros(270)
    angles[90:180] = a_0
    angles[0:90] = a_l90 - 90
    angles[180:270] = a_r90 + 90

    # smooth angles using savitzky golay
    angles_s = savgol_filter(angles, 71, 3)

    plt.plot(angles)
    plt.plot(angles_s)
    plt.show()

    np.savetxt("angles.txt", angles_s)


if __name__ == "__main__":

    # get_and_save_data()
    calib_and_save()
