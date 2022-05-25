import numpy as np

from tianshou.env.multiagent.core import World, Agent, Landmark
from tianshou.env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self,**kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def make_world(self):
        world = World()
        world.max_steps = 25
        world.contained = False

        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.params['num_good_agents']
        num_adversaries = self.params['num_adversaries']
        assert (num_good_agents > 0), 'invalid num of good agents'
        assert (num_adversaries > 0), 'invalid num of adversaries'
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        self.world_radius = 1
        world.obs_radius = self.params['obs_radius'] if 'obs_radius' in self.params else 2 * self.world_radius
        world.rew_shaping = self.params['rew_shape']

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        self.circ_radius = self.world_radius - max([agt.size for agt in world.agents if agt.adversary])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0

        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = (np.array([0.35, 0.85, 0.35]) if not agent.adversary
            #                else np.array([0.85, 0.35, 0.35]))
            agent.color = (np.array([0.85, 0.35, 0.35]) if not agent.adversary
                           else  np.array([0.35, 0.35, 0.85]))

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            if self.params['amb_init']:
                agent.state.p_pos = np.zeros(world.dim_p)
                if agent.adversary:
                    agent.state.p_pos[0] = np.random.uniform(-self.circ_radius, self.circ_radius, 1)
                    agent.state.p_pos[1] = np.sqrt(self.circ_radius ** 2 - agent.state.p_pos[0] ** 2)
                    agent.state.p_pos[1] *= (1 if np.random.uniform(0, 1, 1) > 0.5 else -1)
            else:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            rew = self.adversary_reward(agent, world)
            return [collisions, 0, rew, 0.0, 0.0]
        else:
            # rew = self.agent_reward(agent, world)
            rew = 0.0
            shape = world.rew_shaping
            adversaries = self.adversaries(world)
            if shape:
                # reward can optionally be shaped
                # (increased reward for increased distance from adversary)
                for adv in adversaries:
                    rew += 0.1 * np.sqrt(np.sum(np.square(
                        agent.state.p_pos - adv.state.p_pos)))
            if agent.collide:
                for a in adversaries:
                    if self.is_collision(a, agent):
                        rew -= 10

            # agents are penalized for exiting the screen,
            # so that they can be caught by the adversaries
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
            adversaries = self.adversaries(world)
            min_cr_dist = min([np.sqrt(np.sum(np.square(
                    agent.state.p_pos - adv.state.p_pos))) for adv in adversaries])
            return [0, min_cr_dist, rew, rew-bound_rew, bound_rew]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def good_agents(self, world):
        """Return all agents that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """Return all adversarial agents.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        """Agents are rewarded based on minimum agent distance to each landmark.
        """
        main_reward = (self.adversary_reward(agent, world) if agent.adversary
                       else self.agent_reward(agent, world))
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0.0
        shape = world.rew_shaping
        adversaries = self.adversaries(world)
        if shape:
            # reward can optionally be shaped
            # (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(
                    agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen,
        # so that they can be caught by the adversaries
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

        # Adversaries are rewarded for collisions with agents
        rew = 0.0
        shape = world.rew_shaping
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            # reward can optionally be shaped
            # (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(
                    a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def distance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: [18] np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        [10-11] agent_i's relative location
        [12-13] agent_j's relative location
        Note that i < j
        """
        # get positions of all entities in this agent's reference frame
        lm_vis_mask = []
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            dist = self.distance(agent.state.p_pos, entity.state.p_pos)
            if dist > world.obs_radius:
                lm_vis_mask.append(0)
                entity_pos.append([0,0])
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
                agent_pos.append([0,0])
                agent_vel.append([0,0])
            else:
                agent_vis_mask.append(1)
                agent_pos.append(a.state.p_pos)
                agent_vel.append(a.state.p_vel)

        obs = np.concatenate(
                [agent_vis_mask] +
                [lm_vis_mask] +
                agent_pos +
                agent_vel +
                entity_pos
                )
        return obs

    def done(self, agent, world):
        return world.num_steps >= world.max_steps
