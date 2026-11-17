"""
MAPPO (Multi-Agent PPO) for the disaster response environment.

Structure (CTDE = centralized training, decentralized execution):
  - Three ACTORS (rescue, medical, logistics). Each sees only its own observation
    and picks a zone for every unit it controls.
  - ONE CENTRAL CRITIC. It sees the true global state (env.state()) and estimates how
    good the team's situation is. It is only used during training.
  - All agents share one team reward, so they share one advantage estimate.

Run:   python mappo_agent.py --updates 300 --noise 0.0
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from disaster_pz import DisasterEnv, UNITS_PER_AGENT, NUM_CHOICES_PER_UNIT
from baselines import random_policy, greedy_policy, evaluate_policy

# ---------------- training settings (tune these) ----------------
DISCOUNT = 0.99                # how much future reward matters (gamma)
GAE_LAMBDA = 0.95              # smoothing used when estimating advantages
CLIP_RANGE = 0.2               # PPO: how far a policy may change in one update
LEARNING_RATE = 3e-4
ENTROPY_BONUS = 0.01           # encourages exploration
VALUE_LOSS_WEIGHT = 0.5
MAX_GRADIENT_NORM = 0.5
EPISODES_PER_UPDATE = 32       # episodes collected before each learning step
EPOCHS_PER_UPDATE = 4          # passes over the collected data
MINIBATCH_SIZE = 128
REWARD_SCALE = 0.1             # shrink rewards so the critic learns more easily
HIDDEN_SIZE = 128
TRAINING_FIRST_SEED = 1_000_000   # evaluation uses seeds from 10_000, so no overlap


def build_network(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, HIDDEN_SIZE), nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, output_size),
    )


class Actor(nn.Module):
    """One per agent. Looks at that agent's observation, outputs a choice
    (zone 0-3 or idle) for each unit the agent controls."""

    def __init__(self, observation_size, num_units):
        super().__init__()
        self.num_units = num_units
        self.network = build_network(observation_size, num_units * NUM_CHOICES_PER_UNIT)

    def action_distribution(self, observations):
        scores = self.network(observations)
        scores = scores.view(-1, self.num_units, NUM_CHOICES_PER_UNIT)
        return Categorical(logits=scores)       # one independent choice per unit


class CentralCritic(nn.Module):
    """Looks at the true global state and estimates the team's future reward."""

    def __init__(self, state_size):
        super().__init__()
        self.network = build_network(state_size, 1)

    def forward(self, states):
        return self.network(states).squeeze(-1)


class MAPPO:
    def __init__(self, env):
        self.agent_names = list(env.possible_agents)
        env.reset(seed=0)                        # state() only works after a reset
        state_size = env.state().shape[0]

        self.actors = {
            agent: Actor(env.observation_space(agent).shape[0], UNITS_PER_AGENT[agent])
            for agent in self.agent_names
        }
        self.critic = CentralCritic(state_size)

        self.all_parameters = list(self.critic.parameters())
        for actor in self.actors.values():
            self.all_parameters += list(actor.parameters())
        self.optimizer = torch.optim.Adam(self.all_parameters, lr=LEARNING_RATE)

    # ---------------- 1. play episodes and record everything ----------------
    def collect_episodes(self, env, num_episodes, first_seed):
        names = self.agent_names
        first_agent = names[0]                   # reward and done flags are the same for all
        batch = {
            "states": [], "advantages": [], "returns": [],
            "observations": {a: [] for a in names},
            "actions": {a: [] for a in names},
            "log_probs": {a: [] for a in names},
        }
        episode_scores = []

        for episode_number in range(num_episodes):
            observations, _ = env.reset(seed=first_seed + episode_number)
            states, values, rewards, dones = [], [], [], []
            episode_observations = {a: [] for a in names}
            episode_actions = {a: [] for a in names}
            episode_log_probs = {a: [] for a in names}

            while env.agents:
                state = torch.as_tensor(env.state(), dtype=torch.float32)
                chosen_actions = {}
                with torch.no_grad():
                    value = self.critic(state.unsqueeze(0)).item()
                    for agent in names:
                        observation = torch.as_tensor(observations[agent], dtype=torch.float32)
                        distribution = self.actors[agent].action_distribution(observation.unsqueeze(0))
                        action = distribution.sample()                       # shape (1, units)
                        episode_observations[agent].append(observation)
                        episode_actions[agent].append(action.squeeze(0))
                        episode_log_probs[agent].append(distribution.log_prob(action).sum(-1).item())
                        chosen_actions[agent] = action.squeeze(0).numpy()

                observations, step_rewards, terminated, truncated, _ = env.step(chosen_actions)
                states.append(state)
                values.append(value)
                rewards.append(step_rewards[first_agent] * REWARD_SCALE)
                dones.append(float(terminated[first_agent] or truncated[first_agent]))

            advantages, returns = self.compute_advantages(rewards, values, dones)
            episode_scores.append(sum(rewards) / REWARD_SCALE)

            batch["states"] += states
            batch["advantages"] += list(advantages)
            batch["returns"] += list(returns)
            for agent in names:
                batch["observations"][agent] += episode_observations[agent]
                batch["actions"][agent] += episode_actions[agent]
                batch["log_probs"][agent] += episode_log_probs[agent]

        # turn the python lists into torch tensors
        batch["states"] = torch.stack(batch["states"])
        batch["advantages"] = torch.as_tensor(np.array(batch["advantages"]), dtype=torch.float32)
        batch["returns"] = torch.as_tensor(np.array(batch["returns"]), dtype=torch.float32)
        for agent in names:
            batch["observations"][agent] = torch.stack(batch["observations"][agent])
            batch["actions"][agent] = torch.stack(batch["actions"][agent])
            batch["log_probs"][agent] = torch.as_tensor(
                np.array(batch["log_probs"][agent]), dtype=torch.float32)
        return batch, float(np.mean(episode_scores))

    @staticmethod
    def compute_advantages(rewards, values, dones):
        """Advantage = how much better a step turned out than the critic expected."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        running_advantage = 0.0
        for step in reversed(range(len(rewards))):
            not_done = 1.0 - dones[step]
            next_value = values[step + 1] if step + 1 < len(values) else 0.0
            surprise = rewards[step] + DISCOUNT * next_value * not_done - values[step]
            running_advantage = surprise + DISCOUNT * GAE_LAMBDA * not_done * running_advantage
            advantages[step] = running_advantage
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ---------------- 2. learn from the recorded episodes ----------------
    def update(self, batch):
        num_samples = len(batch["advantages"])
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(EPOCHS_PER_UPDATE):
            shuffled = torch.randperm(num_samples)
            for start in range(0, num_samples, MINIBATCH_SIZE):
                chosen = shuffled[start:start + MINIBATCH_SIZE]
                total_actor_loss = 0.0

                for agent in self.agent_names:
                    distribution = self.actors[agent].action_distribution(
                        batch["observations"][agent][chosen])
                    new_log_prob = distribution.log_prob(batch["actions"][agent][chosen]).sum(-1)
                    ratio = torch.exp(new_log_prob - batch["log_probs"][agent][chosen])
                    unclipped = ratio * advantages[chosen]
                    clipped = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages[chosen]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    total_actor_loss = total_actor_loss + policy_loss - ENTROPY_BONUS * entropy

                predicted_values = self.critic(batch["states"][chosen])
                value_loss = ((predicted_values - batch["returns"][chosen]) ** 2).mean()

                loss = total_actor_loss + VALUE_LOSS_WEIGHT * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_parameters, MAX_GRADIENT_NORM)
                self.optimizer.step()

    # ---------------- 3. use the trained agents ----------------
    def make_policy(self):
        """Policy for evaluation: every unit takes its most likely choice (no randomness)."""
        def policy(env, observations):
            actions = {}
            with torch.no_grad():
                for agent in env.agents:
                    observation = torch.as_tensor(observations[agent], dtype=torch.float32)
                    distribution = self.actors[agent].action_distribution(observation.unsqueeze(0))
                    actions[agent] = distribution.logits.argmax(dim=-1).squeeze(0).numpy()
            return actions
        return policy

    def save(self, path):
        torch.save({"critic": self.critic.state_dict(),
                    **{agent: actor.state_dict() for agent, actor in self.actors.items()}}, path)


def train(num_updates, perception_noise, evaluate_every, save_path, log_path):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    env = DisasterEnv(perception_noise=perception_noise)
    team = MAPPO(env)

    print(f"perception noise = {perception_noise}")
    for name, policy in (("random", random_policy), ("greedy", greedy_policy)):
        score = evaluate_policy(policy, perception_noise=perception_noise)
        print(f"  baseline {name:7s} return {score['mean_return']:7.2f}   deaths {score['mean_deaths']:5.1f}")

    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["update", "training_return", "eval_return", "eval_deaths"])
        next_seed = TRAINING_FIRST_SEED

        for update in range(1, num_updates + 1):
            batch, training_return = team.collect_episodes(env, EPISODES_PER_UPDATE, next_seed)
            next_seed += EPISODES_PER_UPDATE
            team.update(batch)

            if update % evaluate_every == 0 or update == num_updates:
                score = evaluate_policy(team.make_policy(), perception_noise=perception_noise)
                print(f"update {update:4d} | training return {training_return:7.2f} | "
                      f"eval return {score['mean_return']:7.2f} | deaths {score['mean_deaths']:5.1f}")
                log.writerow([update, training_return, score["mean_return"], score["mean_deaths"]])
                log_file.flush()

    team.save(save_path)
    print(f"saved trained agents to {save_path}, learning curve to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.0, help="perception noise (0 = perfect)")
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--save", default="mappo_checkpoint.pt")
    parser.add_argument("--log", default="training_log.csv")
    arguments = parser.parse_args()
    train(arguments.updates, arguments.noise, arguments.evaluate_every, arguments.save, arguments.log)
