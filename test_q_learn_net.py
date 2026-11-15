import numpy as np
import pytest

from action_mapping import MAZE_ACTIONS
from maze_env import Environment
from q_net import GoalNormalizedEncoder, OneHotEncoder, learn, walk


@pytest.mark.parametrize(
    "goal",
    [
        (4, 4),
        (0, 0),
        (2, 2),
    ],
)
def test_q_net_reaches_goal_one_hot(maze, goal):
    env = Environment(maze, goal)
    enc = OneHotEncoder(maze.shape)

    q_func = learn(env, MAZE_ACTIONS, enc)

    path = walk(env, q_func, MAZE_ACTIONS, enc)
    assert path[-2:] == [env.goal, env.goal]


@pytest.mark.parametrize(
    "goal",
    [
        (4, 4),
        (0, 0),
        (2, 2),
    ],
)
def test_q_net_reaches_goal_goal_relative(maze, goal):
    env = Environment(maze, goal)
    enc = GoalNormalizedEncoder(maze.shape)

    q_func = learn(env, MAZE_ACTIONS, enc)

    path = walk(env, q_func, MAZE_ACTIONS, enc)
    assert path[-2:] == [env.goal, env.goal]


def test_goal_relative_generalizes_to_unseen_goals(maze):
    held_out = [(0, 4), (2, 2), (4, 1)]
    start = (0, 0)
    train_goals = [
        cell for cell in np.ndindex(maze.shape) if cell not in held_out and cell != start
    ]
    enc = GoalNormalizedEncoder(maze.shape)

    env = Environment(maze, goals=train_goals, seed=0)

    q_func = learn(env, MAZE_ACTIONS, enc, seed=0)

    for goal in held_out:
        path = walk(Environment(maze, goal), q_func, MAZE_ACTIONS, enc)
        assert path[-2:] == [goal, goal]
