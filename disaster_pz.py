"""
Disaster response environment (PettingZoo ParallelEnv).

Agents : rescue (3 teams), medical (3 ambulances), logistics (2 supply vehicles)
Actions: each unit picks Zone 0..3, or IDLE_ACTION (= 4, stay at base)
Observation (per agent): ALL zone information (noisy, depends on confidence;
                         includes the damage extent of the zone and how long
                         it has waited for a rescue team)
                         + that agent's OWN resources
State (for the centralized critic): the TRUE global situation, via state()
"""
import functools
import numpy as np

# PettingZoo / Gymnasium are only needed for the PettingZoo interface.
# MARLlib (which uses an older gym) can run the simulator without them.
try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError:
    spaces = None
    ParallelEnv = object

# ---------------- basic setup ----------------
NUM_ZONES = 4
IDLE_ACTION = NUM_ZONES                      # action number 4 = "stay at base"
NUM_CHOICES_PER_UNIT = NUM_ZONES + 1         # 4 zones + idle
UNITS_PER_AGENT = {"rescue": 3, "medical": 3, "logistics": 2}
MAX_STEPS = 12                               # 12 steps x 15 minutes = 3 hours
MAX_PEOPLE_PER_ZONE = 60.0                   # used to scale counts into 0-1
INITIAL_SUPPLY_STOCK = 100.0
FEATURES_PER_ZONE = 8                        # trapped, injured, food, road, severity, confidence, waiting, damage

# severity = weighted mix of the three needs (weights are a starting guess)
SEVERITY_WEIGHT_TRAPPED = 0.4
SEVERITY_WEIGHT_INJURED = 0.4
SEVERITY_WEIGHT_FOOD = 0.2

# ---------------- how strong each unit is (tune these) ----------------
PEOPLE_RESCUED_PER_TEAM_PER_STEP = 6.0
PATIENTS_MOVED_PER_AMBULANCE_PER_STEP = 5.0
SUPPLY_DELIVERED_PER_VEHICLE_PER_STEP = 8.0     # people given food/water per truck per step

# Roads blocked by the disaster STAY blocked: nothing in the simulation clears them.
# Ambulances and supply trucks cannot enter a fully blocked zone. Rescue teams still can
# (helicopter, boat, on foot) and work at the SAME speed as everywhere else. In a blocked
# zone they evacuate the injured as well as the trapped, since nobody else can reach them.
# Lower this (for example to 0.5) to make blocked zones harder for rescue teams too.
RESCUE_SPEED_WHEN_ROAD_BLOCKED = 1.0

# Supply trucks cannot drive into a fully blocked zone, but supplies can be air-dropped.
# In a blocked zone a truck's supplies are dropped from the air at this fraction of normal speed.
AIR_DROP_SPEED_WHEN_ROAD_BLOCKED = 0.25

# ---------------- damage extent (how much of a zone the disaster hit) ----------------
# 0 = zone untouched, 1 = zone fully hit (for a flood: fully under water).
# It comes from the image model. A fully hit zone has many trapped people, and waiting is
# more dangerous there. A blocked road with LOW damage means "cut off but safe": supplies
# are needed there, not rescue.
PEOPLE_AT_RISK_AT_FULL_DAMAGE = 55       # trapped people at the start in a fully hit zone
CHANCE_ROAD_BLOCKED_NO_DAMAGE = 0.05     # chance a road is fully blocked when damage is 0 ...
CHANCE_ROAD_BLOCKED_FULL_DAMAGE = 0.45   # ... and when damage is 1 (in between: scales with damage)
CHANCE_ROAD_PARTLY_BLOCKED = 0.25
DAMAGE_INCREASE_WHEN_ZONE_GETS_WORSE = 0.15

# ---------------- how dangerous waiting is (tune these) ----------------
# Each step, every trapped person still waiting has a chance of dying.
# The chance is higher where the disaster hit hardest (blocked roads), and it keeps
# growing the longer a zone goes without any rescue team.
DEATH_CHANCE_OPEN_ROAD = 0.03
DEATH_CHANCE_PARTLY_BLOCKED = 0.06
DEATH_CHANCE_BLOCKED_ROAD = 0.12
DEATH_CHANCE_EXTRA_AT_FULL_DAMAGE = 0.06        # added on top, scaled by the zone's damage extent
DEATH_CHANCE_INCREASE_PER_WAITING_STEP = 0.25   # +25% of the base chance per step with no rescue team
MAX_WAITING_STEPS_COUNTED = 4                   # the increase stops after 4 steps (so at most 2x)


def death_chance(road_access, steps_without_rescue, damage_extent=0.0):
    """Chance that one trapped person dies in one step. Works for one zone or all zones."""
    base_chance = np.where(road_access == 0, DEATH_CHANCE_BLOCKED_ROAD,
                           np.where(road_access < 1, DEATH_CHANCE_PARTLY_BLOCKED,
                                    DEATH_CHANCE_OPEN_ROAD))
    base_chance = base_chance + DEATH_CHANCE_EXTRA_AT_FULL_DAMAGE * damage_extent
    waiting_steps = np.minimum(steps_without_rescue, MAX_WAITING_STEPS_COUNTED)
    return base_chance * (1 + DEATH_CHANCE_INCREASE_PER_WAITING_STEP * waiting_steps)


def supply_speed(road_access):
    """How fast a supply truck works in a zone (1 = normal). Blocked zones get air-drops."""
    return np.where(road_access > 0, road_access, AIR_DROP_SPEED_WHEN_ROAD_BLOCKED)


# ---------------- reward numbers (tune these) ----------------
# Every person helped counts the same, whether rescued, taken to hospital, or given supplies.
REWARD_PER_PERSON_HELPED = 1.0
PENALTY_WASTED_ACTION = -1.0
PENALTY_SWITCHING_ZONE = -0.25
PENALTY_PER_DEATH = -2.0
PENALTY_PER_TRAPPED_PERSON_WAITING = -0.05   # applied every step


class DisasterEnv(ParallelEnv):
    metadata = {"name": "disaster_v0"}

    def __init__(self, perception_noise=0.0):
        # perception_noise = 0 means the agents see the true values.
        # Higher values make the observed numbers noisier in low-confidence zones.
        self.perception_noise = perception_noise
        self.possible_agents = list(UNITS_PER_AGENT)
        self.agents = []
        self.random_generator = np.random.default_rng()

    # ---------------- observation and action spaces ----------------
    def _observation_size(self, agent):
        # extra numbers only some agents have: hospital capacities / supply stock
        own_resource_numbers = {"rescue": 0, "medical": 2, "logistics": 1}[agent]
        unit_location_numbers = NUM_CHOICES_PER_UNIT * UNITS_PER_AGENT[agent]
        time_number = 1
        return (NUM_ZONES * FEATURES_PER_ZONE
                + own_resource_numbers + unit_location_numbers + time_number)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, (self._observation_size(agent),), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # one choice (zone 0-3 or idle) for every unit the agent controls
        return spaces.MultiDiscrete([NUM_CHOICES_PER_UNIT] * UNITS_PER_AGENT[agent])

    # ---------------- helper functions ----------------
    @staticmethod
    def _compute_severity(trapped_scaled, injured_scaled, food_scaled):
        return (SEVERITY_WEIGHT_TRAPPED * trapped_scaled
                + SEVERITY_WEIGHT_INJURED * injured_scaled
                + SEVERITY_WEIGHT_FOOD * food_scaled)

    def _waiting_scaled(self):
        """How long each zone has gone without a rescue team, scaled to 0-1."""
        return np.minimum(self.steps_without_rescue, MAX_WAITING_STEPS_COUNTED) / MAX_WAITING_STEPS_COUNTED

    def _perceived_zone_features(self):
        """What the perception pipeline reports: true values plus noise.
        Zones with low confidence get more noise."""
        noise_size = self.perception_noise * (1 - self.confidence)

        def add_perception_noise(count):
            scaled = count / MAX_PEOPLE_PER_ZONE
            noisy = scaled + self.random_generator.normal(0, noise_size)
            return np.clip(noisy, 0, 1)

        trapped = add_perception_noise(self.trapped_people)
        injured = add_perception_noise(self.injured_people)
        food = add_perception_noise(self.food_need)
        severity = self._compute_severity(trapped, injured, food)
        damage = np.clip(self.damage_extent
                         + self.random_generator.normal(0, noise_size), 0, 1)   # read from images
        # one row per zone, then flattened into a single list of numbers
        waiting = self._waiting_scaled()             # exact: it is bookkeeping, not perception
        return np.stack([trapped, injured, food, self.road_access,
                         severity, self.confidence, waiting, damage], axis=1).ravel()

    def _build_observations(self):
        zone_features = self._perceived_zone_features()   # all agents share this
        fraction_of_time_used = np.array([self.current_step / MAX_STEPS])
        observations = {}
        for agent in self.agents:
            # one-hot: where each of my units currently is
            my_unit_locations = np.eye(NUM_CHOICES_PER_UNIT)[self.unit_locations[agent]].ravel()
            my_resources = [my_unit_locations]
            if agent == "medical":
                my_resources.insert(0, self.hospital_capacity)
            if agent == "logistics":
                my_resources.insert(0, np.array([self.supply_stock / INITIAL_SUPPLY_STOCK]))
            observations[agent] = np.concatenate(
                [zone_features, *my_resources, fraction_of_time_used]).astype(np.float32)
        return observations

    def state(self):
        """TRUE global state for the centralized critic. Never noisy, includes deaths.
        Only used during training, never given to the agents."""
        def scale(count):
            return np.clip(count / MAX_PEOPLE_PER_ZONE, 0, 1)

        trapped = scale(self.trapped_people)
        injured = scale(self.injured_people)
        food = scale(self.food_need)
        zone_features = np.stack(
            [trapped, injured, food, self.road_access,
             self._compute_severity(trapped, injured, food), self.confidence,
             self._waiting_scaled(), self.damage_extent],
            axis=1).ravel()
        all_unit_locations = np.concatenate(
            [np.eye(NUM_CHOICES_PER_UNIT)[self.unit_locations[agent]].ravel()
             for agent in UNITS_PER_AGENT])
        return np.concatenate([
            zone_features,
            scale(self.dead_people),
            self.hospital_capacity,
            [self.supply_stock / INITIAL_SUPPLY_STOCK],
            all_unit_locations,
            [self.current_step / MAX_STEPS],
        ]).astype(np.float32)

    # ---------------- reset: start a new random disaster ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)
        rng = self.random_generator

        self.agents = list(self.possible_agents)
        self.current_step = 0

        # true situation in each zone
        # damage extent: how much of the zone the disaster hit (0 = untouched, 1 = fully hit)
        self.damage_extent = rng.uniform(0.0, 1.0, NUM_ZONES)
        # the more damaged a zone, the more people are trapped and injured
        self.trapped_people = np.round(
            self.damage_extent * rng.uniform(0.6, 1.0, NUM_ZONES) * PEOPLE_AT_RISK_AT_FULL_DAMAGE)
        self.injured_people = np.round(
            self.damage_extent * rng.integers(15, 45, NUM_ZONES) + rng.integers(0, 6, NUM_ZONES))
        self.food_need = rng.integers(5, 45, NUM_ZONES).astype(float)   # people needing food/water
        self.dead_people = np.zeros(NUM_ZONES)
        self.steps_without_rescue = np.zeros(NUM_ZONES)   # steps since a rescue team last went there
        # road access: 1 = open, 0.5 = partly blocked, 0 = fully blocked.
        # Heavily damaged zones are more likely to be cut off, but a zone can be cut off with little
        # damage ("cut off but safe"). Blocked roads STAY blocked (see RESCUE_SPEED_WHEN_ROAD_BLOCKED)
        chance_blocked = (CHANCE_ROAD_BLOCKED_NO_DAMAGE
                          + (CHANCE_ROAD_BLOCKED_FULL_DAMAGE - CHANCE_ROAD_BLOCKED_NO_DAMAGE)
                          * self.damage_extent)
        roll = rng.random(NUM_ZONES)
        self.road_access = np.where(roll < chance_blocked, 0.0,
                                    np.where(roll < chance_blocked + CHANCE_ROAD_PARTLY_BLOCKED, 0.5, 1.0))
        # how sure the perception system is about each zone
        self.confidence = rng.uniform(0.5, 1.0, NUM_ZONES)

        # resources
        self.hospital_capacity = np.ones(2)          # 2 hospitals, 1.0 = empty, 0 = full
        self.supply_stock = INITIAL_SUPPLY_STOCK
        self.unit_locations = {agent: np.full(count, IDLE_ACTION)
                               for agent, count in UNITS_PER_AGENT.items()}

        return self._build_observations(), {agent: {} for agent in self.agents}

    # ---------------- how the disaster changes after agents act ----------------
    def _random_events(self):
        rng = self.random_generator

        if rng.random() < 0.15:                      # a road becomes blocked
            self.road_access[rng.integers(NUM_ZONES)] = 0.0

        if rng.random() < 0.20:                      # one zone gets worse
            worse_zone = rng.integers(NUM_ZONES)
            self.trapped_people[worse_zone] += rng.integers(5, 11)
            self.injured_people[worse_zone] += rng.integers(5, 11)
            self.damage_extent[worse_zone] = min(
                1.0, self.damage_extent[worse_zone] + DAMAGE_INCREASE_WHEN_ZONE_GETS_WORSE)

        self.food_need += rng.integers(1, 4, NUM_ZONES)   # needs keep growing
        self.hospital_capacity = np.clip(
            self.hospital_capacity - rng.uniform(0, 0.05, 2), 0, 1)  # hospitals fill up

    def _trapped_people_die(self):
        """Trapped people still waiting can die. The chance depends on the zone's road
        state, its damage extent and on how long the zone has gone without a rescue team."""
        chance = death_chance(self.road_access, self.steps_without_rescue, self.damage_extent)
        new_deaths = self.random_generator.binomial(self.trapped_people.astype(int), chance)
        self.trapped_people -= new_deaths
        self.dead_people += new_deaths
        return float(new_deaths.sum())

    # ---------------- step: agents act, world responds ----------------
    def step(self, actions):
        team_reward = 0.0

        # --- rescue teams: save trapped people. In a fully BLOCKED zone no ambulance can get in,
        #     so the team also evacuates the injured there (trapped people first). ---
        for zone in actions["rescue"]:
            if zone == IDLE_ACTION:
                continue
            road_is_blocked = self.road_access[zone] == 0
            speed = RESCUE_SPEED_WHEN_ROAD_BLOCKED if road_is_blocked else 1.0
            capacity = PEOPLE_RESCUED_PER_TEAM_PER_STEP * speed     # people this team can move

            rescued = min(self.trapped_people[zone], capacity)
            self.trapped_people[zone] -= rescued
            evacuated_injured = 0.0
            if road_is_blocked:
                evacuated_injured = min(self.injured_people[zone], capacity - rescued)
                self.injured_people[zone] -= evacuated_injured

            people_helped = rescued + evacuated_injured
            if people_helped > 0:
                team_reward += REWARD_PER_PERSON_HELPED * people_helped
            else:
                team_reward += PENALTY_WASTED_ACTION

        # --- ambulances: need a usable road and free hospital space ---
        for zone in actions["medical"]:
            if zone == IDLE_ACTION:
                continue
            best_hospital = int(np.argmax(self.hospital_capacity))
            hospital_has_space = self.hospital_capacity[best_hospital] > 0.05
            if hospital_has_space:
                # half-blocked road means half as many patients moved; blocked means none
                patients_moved = min(
                    self.injured_people[zone],
                    PATIENTS_MOVED_PER_AMBULANCE_PER_STEP * self.road_access[zone])
            else:
                patients_moved = 0.0

            if patients_moved > 0:
                self.injured_people[zone] -= patients_moved
                self.hospital_capacity[best_hospital] = max(
                    0.0, self.hospital_capacity[best_hospital] - 0.02 * patients_moved)
                team_reward += REWARD_PER_PERSON_HELPED * patients_moved
            else:
                team_reward += PENALTY_WASTED_ACTION

        # --- supply trucks: normal speed on open roads, air-drops (slower) into blocked zones ---
        for zone in actions["logistics"]:
            if zone == IDLE_ACTION:
                continue
            supply_delivered = min(
                self.food_need[zone],
                SUPPLY_DELIVERED_PER_VEHICLE_PER_STEP * supply_speed(self.road_access[zone]),
                self.supply_stock)
            if supply_delivered > 0:
                self.food_need[zone] -= supply_delivered
                self.supply_stock -= supply_delivered
                team_reward += REWARD_PER_PERSON_HELPED * supply_delivered
            else:
                team_reward += PENALTY_WASTED_ACTION

        # --- small penalty for switching a unit from one zone to another ---
        for agent, chosen_zones in actions.items():
            for unit_index, new_zone in enumerate(chosen_zones):
                previous_zone = self.unit_locations[agent][unit_index]
                if (new_zone != IDLE_ACTION and previous_zone != IDLE_ACTION
                        and new_zone != previous_zone):
                    team_reward += PENALTY_SWITCHING_ZONE
            self.unit_locations[agent] = np.asarray(chosen_zones, dtype=int)

        # --- zones a rescue team went to are no longer "waiting"; the others wait one more step ---
        zones_with_a_rescue_team = set(int(zone) for zone in actions["rescue"]) - {IDLE_ACTION}
        for zone in range(NUM_ZONES):
            if zone in zones_with_a_rescue_team:
                self.steps_without_rescue[zone] = 0
            else:
                self.steps_without_rescue[zone] += 1

        # --- the disaster changes AFTER the agents have acted ---
        self._random_events()
        num_deaths = self._trapped_people_die()
        team_reward += PENALTY_PER_DEATH * num_deaths
        team_reward += PENALTY_PER_TRAPPED_PERSON_WAITING * self.trapped_people.sum()

        self.current_step += 1
        # The episode ends early only when nobody is trapped or injured AND the food need is
        # finished (or the supply stock has run out, so nothing more can be delivered).
        food_is_done = bool(self.food_need.sum() == 0 or self.supply_stock <= 0)
        everyone_helped = bool(self.trapped_people.sum() + self.injured_people.sum() == 0
                               and food_is_done)
        out_of_time = self.current_step >= MAX_STEPS

        observations = self._build_observations()    # build before clearing agents
        result = (
            observations,
            {agent: float(team_reward) for agent in self.agents},   # shared team reward
            {agent: everyone_helped for agent in self.agents},
            {agent: out_of_time for agent in self.agents},
            {agent: {"total_dead": float(self.dead_people.sum())} for agent in self.agents},
        )
        if everyone_helped or out_of_time:
            self.agents = []
        return result


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    parallel_api_test(DisasterEnv(perception_noise=0.3), num_cycles=100)
    print("API test passed")

    env = DisasterEnv(perception_noise=0.3)
    observations, _ = env.reset(seed=0)
    print({agent: obs.shape for agent, obs in observations.items()},
          "| critic state:", env.state().shape)

    for episode in range(3):
        observations, _ = env.reset(seed=episode)
        episode_return = 0.0
        while env.agents:
            random_actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminated, truncated, infos = env.step(random_actions)
            episode_return += rewards["rescue"]
        print(f"random episode {episode}: team return {episode_return:.1f}, "
              f"dead {infos['rescue']['total_dead']:.0f}")