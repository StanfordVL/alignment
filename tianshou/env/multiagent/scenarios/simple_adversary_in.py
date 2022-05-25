import numpy as np

from tianshou.env.multiagent.core import World, Agent, Landmark
from tianshou.env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def make_world(self):
        world = World()
        world.max_steps = 25

        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.params['num_good_agents']  # 2
        num_adversaries = self.params['num_adversaries']  # 1
        assert (num_good_agents > 0), 'invalid num of good agents'
        assert (num_adversaries > 0), 'invalid num of adversaries'
        num_agents = num_adversaries + num_good_agents
        world.num_agents = num_agents
        num_landmarks = num_agents - 1
        self.world_radius = 1
        world.obs_radius = self.params['obs_radius']
        world.rew_shaping = self.params['rew_shape']

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        self.circ_radius = self.world_radius - \
            max([landmark.size for landmark in world.landmarks])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0

        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])

        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal

        # set random initial states
        for agent in world.agents:
            if self.params['amb_init']:
                agent.state.p_pos = np.zeros(world.dim_p)
            else:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if self.params['amb_init']:
                landmark.state.p_pos = np.zeros(world.dim_p)
                landmark.state.p_pos[0] = np.random.uniform(
                    -self.circ_radius, self.circ_radius, 1)
                landmark.state.p_pos[1] = np.sqrt(
                    self.circ_radius ** 2 - landmark.state.p_pos[0] ** 2)
                landmark.state.p_pos[1] *= (1 if np.random.uniform(0,
                                                                   1, 1) > 0.5 else -1)
            else:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def distance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        occupied = 0
        if self.is_collision(agent, agent.goal_a):
            occupied += 1
        if agent.adversary:
            # min adv-goal distance
            dists = [self.distance(a.state.p_pos, a.goal_a.state.p_pos)
                     for a in self.adversaries(world)]
            rew = self.adversary_reward(agent, world)
            return rew, occupied, min(dists) / len(self.adversaries(world)), 0.0, 0.0
        else:
            # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
            shaped_reward = world.rew_shaping
            shaped_adv_reward = world.rew_shaping

            # Calculate negative reward for adversary
            adversary_agents = self.adversaries(world)
            if shaped_adv_reward:  # distance-based adversary reward
                adv_rew = sum([np.sqrt(np.sum(
                    np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            else:  # proximity-based adversary reward (binary)
                adv_rew = 0.0
                for a in adversary_agents:
                    if self.is_collision(a, a.goal_a):
                        adv_rew -= 5.0

            # Calculate positive reward for agents
            good_agents = self.good_agents(world)
            if shaped_reward:  # distance-based agent reward
                pos_rew = -min(
                    [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
            else:  # proximity-based agent reward (binary)
                pos_rew = 0.0
                if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                        < agent.goal_a.size + good_agents[0].size:
                    pos_rew += 5.0
                    # pos_rew -= min(
                #     [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
            rew = pos_rew + adv_rew

            def bound(x):
                if x > 1.0:
                    return min(np.exp(2 * x - 2), 10)
                else:
                    return 0.0
            bound_rew = 0.0
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                bound_rew -= bound(x)
            rew += bound_rew
            # min agent-goal distance
            dists = [self.distance(a.state.p_pos, a.goal_a.state.p_pos)
                     for a in self.good_agents(world)]
            # rew = self.agent_reward(agent, world)
            return rew, occupied, min(dists) / len(self.good_agents(world)), pos_rew, bound_rew

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = world.rew_shaping
        shaped_adv_reward = world.rew_shaping

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(
                np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0.0
            for a in adversary_agents:
                if self.is_collision(a, a.goal_a):
                    adv_rew -= 5.0

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0.0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < agent.goal_a.size + good_agents[0].size:
                pos_rew += 5.0
                # pos_rew -= min(
            #     [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        rew = pos_rew + adv_rew

        def bound(x):
            if x > 1.0:
                return min(np.exp(2 * x - 2), 10)
            else:
                return 0.0
        bound_rew = 0.0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            bound_rew -= bound(x)
        rew += bound_rew
        return rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = world.rew_shaping
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0.0
            if self.is_collision(agent.goal_a, agent):
                adv_rew += 5.0
            return adv_rew

    def observation(self, agent, world):
        lm_vis_mask = []
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            dist = self.distance(agent.state.p_pos, entity.state.p_pos)
            if dist > world.obs_radius:
                lm_vis_mask.append(0)
                entity_pos.append([0, 0])
            else:
                lm_vis_mask.append(1)
                entity_pos.append(entity.state.p_pos)
        # communication of all other agents
        agent_pos = []
        agent_vel = []
        agent_vis_mask = []
        for a in world.agents:
            dist = self.distance(agent.state.p_pos, a.state.p_pos)
            if dist > world.obs_radius:
                agent_vis_mask.append(0)
                agent_pos.append([0, 0])
                agent_vel.append([0, 0])
            else:
                agent_vis_mask.append(1)
                agent_pos.append(a.state.p_pos)
                agent_vel.append(a.state.p_vel)

        if not agent.adversary:
            return np.concatenate(
                [agent_vis_mask] +
                [lm_vis_mask] +
                agent_pos +
                agent_vel +
                entity_pos +
                [agent.goal_a.state.p_pos])
        else:
            return np.concatenate(
                [agent_vis_mask] +
                [lm_vis_mask] +
                agent_pos +
                agent_vel +
                entity_pos +
                [np.zeros(2)])

    def done(self, agent, world):
        return world.num_steps >= world.max_steps
