import numpy as np

from action_mapping import MAZE_ACTIONS
from maze_env import Environment


class QFuncTable:
    def __init__(
        self, state_space, action_count, learning_rate=0.1, discount_factor=0.9
    ):
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((state_space[0], state_space[1], action_count))

    def __call__(self, state, action_index):
        x, y = state
        return self.q_table[x, y, action_index]

    def update_q_value(self, state, action_index, reward, next_state, done):
        x, y = state
        next_x, next_y = next_state

        best_next_q_value = np.max(self.q_table[next_x, next_y])
        td_target = reward + self.discount_factor * (1 - done) * best_next_q_value
        td_error = td_target - self.q_table[x, y, action_index]

        self.q_table[x, y, action_index] += self.learning_rate * td_error


class MaxPolicy:
    def __init__(self, q_func: QFuncTable):
        self.q_func = q_func

    def __call__(self, state):
        q_values = [self.q_func(state, i) for i in range(self.q_func.action_count)]
        max_q_value = max(q_values)
        best_actions = [
            i for i, q in enumerate(q_values) if q == max_q_value
        ]
        return np.random.choice(best_actions)


def learn_table(env: Environment, actions):
    state_space = env.maze.shape
    q_func = QFuncTable(state_space, len(actions))
    policy = MaxPolicy(q_func)

    for episode in range(1000):
        state = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < 0.1:
                action_index = np.random.randint(len(actions))
            else:
                action_index = policy(state)

            next_state, reward, done = env.step(actions[action_index])
            q_func.update_q_value(state, action_index, reward, next_state, done)
            state = next_state

    return q_func


def walk(env: Environment, q_func: QFuncTable, actions):
    state = env.reset()
    done = False
    path = [state]
    policy = MaxPolicy(q_func)

    while not done:
        action_index = policy(state)
        next_state, reward, done = env.step(actions[action_index])
        path.append(next_state)
        state = next_state

    return path


if __name__ == "__main__":
    maze = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    goal = (4, 4)
    env = Environment(maze, goal)
    q_func = learn_table(env, MAZE_ACTIONS)

    path = walk(env, q_func, MAZE_ACTIONS)
    assert path[-1] == goal
