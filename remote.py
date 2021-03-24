from typing import Optional
import termios
import sys
import tty
from threading import Thread
from actions import Action


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

    def apply(self, agent) -> Action:
        c = self.last
        if c == "w":  # forward
            return Action.Forward
        elif c == "r":  # back
            return Action.Backward
        elif c == "s":  # left
            return Action.Left
        elif c == "a":  # right
            return Action.Right
        elif c == "q":
            sys.exit("q pressed during remote")
        elif c == "p":
            return Action.Save
        else:
            return Action.Stay


def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
