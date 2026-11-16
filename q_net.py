import torch
import random
from torch import nn, optim
import numpy as np
from collections import deque

from action_mapping import MAZE_ACTIONS
from maze_env import Environment


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_count):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_count)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class OneHotEncoder:
    def __init__(self, shape):
        self._shape = shape

    @property
    def input_dim(self):
        return int(np.prod(self._shape))

    def encode(self, state, goal):
        # the goal is not part of the input, so a net trained with this encoder
        # only knows the goal it was trained on
        one_hot = np.zeros(self._shape, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot.ravel()


class GoalNormalizedEncoder:
    def __init__(self, shape):
        self._ndim = len(shape)
        # one scale for all axes so a step has the same size along each of them
        self.scale = max(max(shape) - 1, 1)

    @property
    def input_dim(self):
        return 2 * self._ndim

    def encode(self, state, goal):
        state = np.asarray(state, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        return np.concatenate([state / self.scale, (goal - state) / self.scale])


def greedy_action(q_net, encoder, state, goal):
    state_tensor = torch.from_numpy(encoder.encode(state, goal)).unsqueeze(0)
    with torch.no_grad():
        return torch.argmax(q_net(state_tensor)).item()


def greedy_reaches_goal(env: Environment, q_net: QNetwork, encoder, action_space):
    state = env.reset()
    visited = {state}
    for _ in range(env.maze.size):
        action_index = greedy_action(q_net, encoder, state, env.goal)
        state, _, done = env.step(action_space[action_index])
        if done:
            return True
        if state in visited:
            return False
        visited.add(state)
    return False


class EGreedyPolicy:
    def __init__(self, q_net, encoder, action_space, epsilon):
        self.q_net = q_net
        self.encoder = encoder
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, state, goal):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(len(self.action_space))
        return greedy_action(self.q_net, self.encoder, state, goal)


def learn(env: Environment, action_space, encoder, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    replay_buffer = deque(maxlen=10000)

    q_net_online = QNetwork(encoder.input_dim, len(action_space))
    q_net_target = QNetwork(encoder.input_dim, len(action_space))

    q_net_target.load_state_dict(q_net_online.state_dict())

    optimizer = optim.Adam(q_net_online.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    max_episodes = 1000
    max_steps = env.maze.size * 4
    gamma = 0.9
    batch_size = 32
    epsilon_min = 0.05
    epsilon_decay = 0.99
    eval_every = 20
    # one successful eval can be luck, and stopping then leaves goals near the
    # training ones undertrained
    evals_to_stop = 5
    successful_evals = 0
    tau = 0.005

    policy = EGreedyPolicy(q_net_online, encoder, action_space, epsilon=0.2)

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        goal = env.goal

        policy.epsilon = max(epsilon_min, epsilon_decay**episode)

        for _ in range(max_steps):
            action_index = policy.select_action(state, goal)

            next_state, reward, done = env.step(action_space[action_index])

            replay_buffer.append(
                (
                    encoder.encode(state, goal),
                    action_index,
                    reward,
                    encoder.encode(next_state, goal),
                    int(done),
                )
            )

            # start training after enough samples are collected
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # y = r + gamma * (1 - done) * max(Q(s', a'))
                with torch.no_grad():
                    target_q_values = (
                        rewards
                        + (1 - dones)
                        * gamma
                        * q_net_target(next_states).max(dim=1).values
                    )

                prediction = q_net_online(states)[torch.arange(len(states)), actions]

                loss = criterion(prediction, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for target_param, online_param in zip(
                        q_net_target.parameters(), q_net_online.parameters()
                    ):
                        target_param.lerp_(online_param, tau)

            state = next_state
            if done:
                break

        if episode % eval_every == 0:
            if _reaches_all_goals(env, q_net_target, encoder, action_space):
                successful_evals += 1
            else:
                successful_evals = 0
            if successful_evals == evals_to_stop:
                return q_net_target

    return q_net_target


def _reaches_all_goals(env: Environment, q_net, encoder, action_space):
    for goal in env.goals:
        eval_env = Environment(env.maze, goal)
        if not greedy_reaches_goal(eval_env, q_net, encoder, action_space):
            return False
    return True


def walk(env: Environment, q_net: QNetwork, actions, encoder):
    state = env.reset()
    path = [state]
    visited = {state}

    while True:
        action_index = greedy_action(q_net, encoder, state, env.goal)

        next_state, reward, done = env.step(actions[action_index])
        path.append(next_state)
        if done:
            return path
        if next_state in visited:
            raise RuntimeError(
                f"Greedy policy repeats state {next_state}; path: {path}"
            )
        visited.add(next_state)
        state = next_state


if __name__ == "__main__":
    maze = np.zeros((5, 5))
    goal = (4, 4)
    env = Environment(maze, goal)
    enc = OneHotEncoder(maze.shape)

    q_net = learn(env, MAZE_ACTIONS, enc)
    path = walk(env, q_net, MAZE_ACTIONS, enc)
    print("Path taken:", path)
