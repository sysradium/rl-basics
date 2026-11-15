import random
from enum import Enum


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


class Environment:
    def __init__(self, maze, goal=None, *, goals=None, seed=None):
        if (goal is None) == (goals is None):
            raise ValueError("Pass exactly one of goal or goals")
        self.maze = maze
        self.state = (0, 0)
        self._goals = (goal,) if goals is None else tuple(goals)
        self._goal = self._goals[0]
        self._rng = random.Random(seed)

    @property
    def goal(self):
        return self._goal

    @property
    def goals(self):
        return self._goals

    def reset(self):
        if len(self._goals) > 1:
            self._goal = self._rng.choice(self._goals)
        self.state = (0, 0)
        return self.state

    def step(self, action: Action):
        x, y = self.state
        match action:
            case Action.UP:
                x = max(0, x - 1)
            case Action.DOWN:
                x = min(len(self.maze) - 1, x + 1)
            case Action.LEFT:
                y = max(0, y - 1)
            case Action.RIGHT:
                y = min(len(self.maze[0]) - 1, y + 1)
            case Action.STAY:
                pass
            case _:
                raise ValueError(f"Unsupported action: {action}")

        self.state = (x, y)
        # arriving at the goal is not enough, the episode ends only once the agent stays there
        if self.state == self._goal and action is Action.STAY:
            return self.state, 100, True

        return self.state, -0.1, False
