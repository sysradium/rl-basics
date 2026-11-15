import pytest

from action_mapping import MAZE_ACTIONS
from maze_env import Environment
from q_learn_table import learn_table, walk


@pytest.mark.parametrize(
    "goal",
    [
        (4, 4),
        (0, 0),
        (2, 2),
    ],
)
def test_q_table_reaches_goal(maze, goal):
    env = Environment(maze, goal)
    q_func = learn_table(env, MAZE_ACTIONS)

    __import__("pdb").set_trace()
    path = walk(env, q_func, MAZE_ACTIONS)
    assert path[-1] == env.goal
