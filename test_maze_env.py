import numpy as np
import pytest

from action_mapping import MAZE_ACTIONS
from maze_env import Action, Environment


def test_action_order_maps_indices_before_environment_step():
    env = Environment(np.zeros((2, 2)), (1, 1))
    actions = (Action.RIGHT, Action.DOWN, Action.UP, Action.LEFT, Action.STAY)

    assert len(MAZE_ACTIONS) == 5
    assert env.step(actions[0]) == ((0, 1), -0.1, False)
    assert env.step(actions[1]) == ((1, 1), -0.1, False)


def test_environment_accepts_actions_directly():
    env = Environment(np.zeros((2, 2)), (1, 1))

    assert env.step(Action.RIGHT) == ((0, 1), -0.1, False)
    assert env.step(Action.DOWN) == ((1, 1), -0.1, False)


def test_reset_samples_goal_from_goals():
    goals = [(0, 1), (1, 0), (1, 1)]
    env = Environment(np.zeros((2, 2)), goals=goals, seed=0)

    sampled = set()
    for _ in range(20):
        env.reset()
        sampled.add(env.goal)

    assert sampled <= set(goals)
    assert len(sampled) > 1


def test_goal_is_kept_across_resets_for_single_goal():
    env = Environment(np.zeros((2, 2)), (1, 1))
    env.reset()

    assert env.goal == (1, 1)


@pytest.mark.parametrize("kwargs", [{}, {"goal": (1, 1), "goals": [(1, 1)]}])
def test_requires_exactly_one_of_goal_or_goals(kwargs):
    with pytest.raises(ValueError):
        Environment(np.zeros((2, 2)), **kwargs)


def test_goal_cannot_be_assigned():
    env = Environment(np.zeros((2, 2)), (1, 1))

    with pytest.raises(AttributeError):
        env.goal = (0, 1)


def test_stay_at_goal_ends_episode():
    env = Environment(np.zeros((2, 2)), (1, 1))
    env.step(Action.DOWN)
    env.step(Action.RIGHT)

    assert env.step(Action.STAY) == ((1, 1), 100, True)


def test_stay_outside_goal_is_a_regular_step():
    env = Environment(np.zeros((2, 2)), (1, 1))

    assert env.step(Action.STAY) == ((0, 0), -0.1, False)


def test_bumping_into_wall_on_goal_does_not_end_episode():
    env = Environment(np.zeros((2, 2)), (1, 1))
    env.step(Action.DOWN)
    env.step(Action.RIGHT)

    assert env.step(Action.DOWN) == ((1, 1), -0.1, False)
