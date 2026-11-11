"""
Simple policies to compare the trained agents against, plus the function that scores any policy.
A "policy" here is any function:  policy(env, observations) -> actions dictionary

  random_policy               every unit picks randomly
  greedy_single_zone_policy   all units of a kind go to the zone with the biggest need (weak)
  greedy_policy               units are assigned one by one, spreading across zones (strong)
"""
import numpy as np
from disaster_pz import (
    DisasterEnv, UNITS_PER_AGENT, NUM_ZONES, FEATURES_PER_ZONE, MAX_PEOPLE_PER_ZONE, IDLE_ACTION,
    PEOPLE_RESCUED_PER_TEAM_PER_STEP, PATIENTS_MOVED_PER_AMBULANCE_PER_STEP,
    SUPPLY_DELIVERED_PER_VEHICLE_PER_STEP, RESCUE_SPEED_WHEN_ROAD_BLOCKED,
    DEATH_CHANCE_OPEN_ROAD, MAX_WAITING_STEPS_COUNTED, death_chance, supply_speed,
)

# position of each feature inside one zone's block of numbers in the observation
TRAPPED, INJURED, FOOD, ROAD_ACCESS, SEVERITY, CONFIDENCE, WAITING, DAMAGE = range(FEATURES_PER_ZONE)


def zone_table(observation):
    """Turn the flat observation into a table: one row per zone, one column per feature."""
    return observation[: NUM_ZONES * FEATURES_PER_ZONE].reshape(NUM_ZONES, FEATURES_PER_ZONE)


def urgency(zones):
    """How many times faster people die in each zone than in a normal open-road zone
    (blocked roads, heavy damage and long waits make a zone more urgent). Rescue teams use
    this to decide where the dire zones are."""
    steps_waiting = zones[:, WAITING] * MAX_WAITING_STEPS_COUNTED
    return death_chance(zones[:, ROAD_ACCESS], steps_waiting, zones[:, DAMAGE]) / DEATH_CHANCE_OPEN_ROAD


def random_policy(env, observations):
    """Every unit picks a random zone (or idle)."""
    return {agent: env.action_space(agent).sample() for agent in env.agents}


def greedy_single_zone_policy(env, observations):
    """Each agent sends ALL its units to the zone where its own kind of need is biggest."""
    actions = {}
    for agent in env.agents:
        zones = zone_table(observations[agent])
        if agent == "rescue":
            need = (zones[:, TRAPPED] + np.where(zones[:, ROAD_ACCESS] == 0, zones[:, INJURED], 0.0)) * urgency(zones)
        elif agent == "medical":
            need = zones[:, INJURED] * zones[:, ROAD_ACCESS]
        else:  # logistics
            need = zones[:, FOOD] * supply_speed(zones[:, ROAD_ACCESS])
        actions[agent] = np.full(UNITS_PER_AGENT[agent], int(np.argmax(need)))
    return actions


def greedy_policy(env, observations):
    """Assign units ONE BY ONE, each to the zone where it would help the most people
    (rescue teams count people in dire zones, such as blocked roads or long waits, for more),
    taking into account what the units before it are already going to do
    (people already covered by an earlier unit are not counted again).
    This leaves a unit idle when
    there is nothing useful left. It knows how fast each unit works, which the learned
    agents have to discover, so it is a fairly strong baseline."""
    actions = {}
    for agent in env.agents:
        zones = zone_table(observations[agent])
        road = zones[:, ROAD_ACCESS]
        zone_weight = np.ones(NUM_ZONES)             # how much each person helped counts
        if agent == "rescue":
            zone_weight = urgency(zones)             # dire zones count for more
            # in a fully blocked zone the rescue team also evacuates the injured
            remaining_need = (zones[:, TRAPPED]
                              + np.where(road == 0, zones[:, INJURED], 0.0)) * MAX_PEOPLE_PER_ZONE
            speed = np.where(road > 0, 1.0, RESCUE_SPEED_WHEN_ROAD_BLOCKED)
            full_rate = PEOPLE_RESCUED_PER_TEAM_PER_STEP * speed
        elif agent == "medical":
            remaining_need = zones[:, INJURED] * MAX_PEOPLE_PER_ZONE
            full_rate = PATIENTS_MOVED_PER_AMBULANCE_PER_STEP * road
        else:  # logistics
            remaining_need = zones[:, FOOD] * MAX_PEOPLE_PER_ZONE
            full_rate = SUPPLY_DELIVERED_PER_VEHICLE_PER_STEP * supply_speed(road)

        choices = []
        for _ in range(UNITS_PER_AGENT[agent]):
            people_helped = np.minimum(remaining_need, full_rate)
            best_zone = int(np.argmax(people_helped * zone_weight + 1e-3 * remaining_need))
            if people_helped[best_zone] < 0.5:                              # nothing useful to do
                choices.append(IDLE_ACTION)
            else:
                choices.append(best_zone)
                remaining_need[best_zone] -= people_helped[best_zone]
        actions[agent] = np.array(choices)
    return actions


def evaluate_policy(policy, num_episodes=200, first_seed=10_000, perception_noise=0.0):
    """Run a policy on the same set of scenarios (seeds) and report the average score.
    Using identical seeds for every policy makes the comparison fair."""
    env = DisasterEnv(perception_noise=perception_noise)
    episode_returns, episode_deaths = [], []
    for episode in range(num_episodes):
        observations, _ = env.reset(seed=first_seed + episode)
        episode_return = 0.0
        while env.agents:
            actions = policy(env, observations)
            observations, rewards, terminated, truncated, infos = env.step(actions)
            episode_return += rewards["rescue"]      # team reward is the same for all agents
        episode_returns.append(episode_return)
        episode_deaths.append(infos["rescue"]["total_dead"])
    return {"mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_deaths": float(np.mean(episode_deaths))}


if __name__ == "__main__":
    policies = (("random", random_policy),
                ("greedy, single zone", greedy_single_zone_policy),
                ("greedy, spread", greedy_policy))
    for noise in (0.0, 0.3):
        print(f"--- perception noise = {noise} ---")
        for name, policy in policies:
            score = evaluate_policy(policy, perception_noise=noise)
            print(f"{name:20s} return {score['mean_return']:7.2f} +/- {score['std_return']:5.2f}"
                  f"   deaths {score['mean_deaths']:5.1f}")