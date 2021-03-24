"""
"""

from __future__ import print_function
from src.env import VrepEnvironment
from src.agents import Pioneer
import settings
import time
import bot

""" Motors:

          1. agent.change_velocity([ speed_left: float, speed_right: float ])
               Set the target angular velocities of left
               and right motors with a LIST of values:
               e.g. [1., 1.] in radians/s.

               Values in range [-5:5] (above these
               values the control accuracy decreases)

          2. agent.current_velocity()
          ----
               Returns a LIST of current angular velocities
               of the motors
               [speed_left: float, speed_right: float] in radians/s.

    Lidar:
          3. agent.read_lidar()
          ----
               Returns a list of floating point numbers that you can indicate
               the distance towards the closest object at a particular angle.

               Basic configuration of the lidar:
               Angle: [-135:135] Starting with the
               leftmost lidar point -> clockwise

    Agent:
          You can access these attributes to get information about the agent's
          positions

          4. agent.pos

          ----
               Current x,y position of the agent (derived from
               SLAM data)

          5. agent.position_history

               A deque containing N last positions of the agent
               (200 by default, can be changed in settings.py)
"""

if __name__ == "__main__":
    # Initialize and start the environment
    # Open the file containing our scene (robot and its environment)
    environment = VrepEnvironment(settings.SCENES + '/room_static.ttt')
    environment.connect()        # Connect python to the simulator's remote API
    agent = Pioneer(environment)

    print('\nDemonstration of Simultaneous Localization and Mapping using '
          'CoppeliaSim robot simulation software. \nPress "CTRL+C" to exit.\n')
    start = time.time()
    step = 0
    done = False
    environment.start_simulation()
    time.sleep(1)
    state = bot.State()

    try:
        while step < settings.simulation_steps and not done:
            # display.update()                      # Update the SLAM display
            bot.loop(agent, state)                           # Control loop
            step += 1
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()-start))

    # display.close()
    environment.stop_simulation()
    environment.disconnect()
