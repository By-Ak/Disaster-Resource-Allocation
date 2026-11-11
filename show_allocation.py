"""
Show, step by step, how the agents allocate resources in ONE disaster scenario.

    python show_allocation.py --model clean.pt --seed 5                 (trained agents)
    python show_allocation.py --model clean.pt --seed 5 --noise 0.3     (with imperfect perception)
    python show_allocation.py --greedy --seed 5                         (simple rule, no model needed)

Change --seed to see a different disaster.
"""
import argparse

from disaster_pz import DisasterEnv, UNITS_PER_AGENT, NUM_ZONES, IDLE_ACTION, MAX_STEPS, MAX_PEOPLE_PER_ZONE
from baselines import greedy_policy, zone_table, TRAPPED, INJURED, FOOD

ZONE_NAMES = ["Zone A", "Zone B", "Zone C", "Zone D"]
UNIT_LABELS = {"rescue": "Rescue team", "medical": "Ambulance", "logistics": "Supply truck"}
ROAD_WORDS = {1.0: "open", 0.5: "partly blocked", 0.0: "BLOCKED"}


def print_situation(env, observations, noise):
    """The state of every zone. With noise, also show what the agents THOUGHT they saw."""
    seen = zone_table(observations["rescue"]) * MAX_PEOPLE_PER_ZONE   # agents' view, as counts

    def cell(true_value, seen_value):
        return f"{true_value:.0f}" if noise == 0 else f"{true_value:.0f} (saw {seen_value:.0f})"

    print(f"  {'Zone':8s}{'Damage':>8s}{'Trapped':>16s}{'Injured':>16s}{'Food need':>16s}   Road")
    for zone in range(NUM_ZONES):
        print(f"  {ZONE_NAMES[zone]:8s}"
              f"{env.damage_extent[zone] * 100:7.0f}%"
              f"{cell(env.trapped_people[zone], seen[zone, TRAPPED]):>16s}"
              f"{cell(env.injured_people[zone], seen[zone, INJURED]):>16s}"
              f"{cell(env.food_need[zone], seen[zone, FOOD]):>16s}"
              f"   {ROAD_WORDS[float(env.road_access[zone])]}")
    print(f"  Hospital space left: {env.hospital_capacity[0]*100:.0f}% and {env.hospital_capacity[1]*100:.0f}%"
          f"  |  Supply stock left: {env.supply_stock:.0f}")


def print_allocation(actions):
    print("  Decisions:")
    for agent, choices in actions.items():
        for unit_number, choice in enumerate(choices, start=1):
            where = "stays idle" if choice == IDLE_ACTION else f"goes to {ZONE_NAMES[choice]}"
            print(f"    {UNIT_LABELS[agent]} {unit_number} {where}")


def run_episode(policy, seed, noise, verbose):
    env = DisasterEnv(perception_noise=noise)
    observations, _ = env.reset(seed=seed)
    total_return, step_number = 0.0, 0
    while env.agents:
        step_number += 1
        if verbose:
            print(f"\n=== Step {step_number}  (minute {15 * (step_number - 1)} of {15 * MAX_STEPS}) ===")
            print_situation(env, observations, noise)
        actions = policy(env, observations)
        if verbose:
            print_allocation(actions)
        observations, rewards, terminated, truncated, infos = env.step(actions)
        total_return += rewards["rescue"]
        if verbose:
            print(f"  Result: team reward {rewards['rescue']:+.1f} | still trapped {env.trapped_people.sum():.0f}"
                  f" | still injured {env.injured_people.sum():.0f} | deaths so far {env.dead_people.sum():.0f}")
    return total_return, float(env.dead_people.sum())


def load_trained_policy(model_path):
    import torch                                   # only needed when loading a trained model
    from mappo_agent import MAPPO
    team = MAPPO(DisasterEnv())
    checkpoint = torch.load(model_path)
    team.critic.load_state_dict(checkpoint["critic"])
    for agent in team.agent_names:
        team.actors[agent].load_state_dict(checkpoint[agent])
    return team.make_policy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="saved agents, e.g. clean.pt")
    parser.add_argument("--greedy", action="store_true", help="show the simple rule instead")
    parser.add_argument("--seed", type=int, default=5, help="which disaster scenario")
    parser.add_argument("--noise", type=float, default=0.0, help="perception noise (0 = perfect)")
    args = parser.parse_args()

    use_greedy = args.greedy or args.model is None
    policy = greedy_policy if use_greedy else load_trained_policy(args.model)
    name = "simple rule (greedy)" if use_greedy else f"trained agents ({args.model})"
    print(f"Scenario seed {args.seed}, perception noise {args.noise}, decisions by: {name}")

    score, deaths = run_episode(policy, args.seed, args.noise, verbose=True)
    print(f"\nFINAL: total return {score:.1f}, deaths {deaths:.0f}")
    if not use_greedy:
        rule_score, rule_deaths = run_episode(greedy_policy, args.seed, args.noise, verbose=False)
        print(f"Same scenario with the simple rule: return {rule_score:.1f}, deaths {rule_deaths:.0f}")