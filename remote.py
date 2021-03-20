from typing import Optional
import termios
import sys
import tty
from threading import Thread


class Control:
    def __init__(self):
        self.last = " "
        self.forward = 0
        self.thread = Thread(target=self.update)
        self.thread.start()

    def update(self):
        while True:
            self.last = get_char()
            if self.last == "q":
                sys.exit("q pressed during remote")

    def apply(self, agent) -> Optional[str]:
        c = self.last
        if c == "w":  # forward
            self.forward = 0.15
            agent.change_velocity([self.forward, self.forward], target=None)
        elif c == "r":  # back
            self.forward = 0.15
            agent.change_velocity([self.forward, self.forward], target=None)
        elif c == "s":  # left
            agent.change_velocity([0.1, -0.1], target=None)
        elif c == "a":  # right
            agent.change_velocity([-0.1, 0.1], target=None)
        elif c == "q":
            sys.exit("q pressed during remote")
        else:
            return c
        return None


def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
