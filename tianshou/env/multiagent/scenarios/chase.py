import numpy as np

from tianshou.env.multiagent.core import World, Agent, Landmark
from tianshou.env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        world.max_steps = 100
        world.contained = True
        world.random_init = True

        # set any world properties first
        world.dim_c = 2
        num_chasers = 2
        num_runners = 1
        num_agents = num_chasers + num_runners
        num_landmarks = 2

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.chaser = True if i < num_chasers else False
            agent.size = 0.075 if agent.chaser else 0.05
            agent.accel = 3.0 if agent.chaser else 4.0
            # agent.accel = 20.0 if agent.chaser else 25.0
            agent.max_speed = 1.0 if agent.chaser else 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0

        # random properties for agents
        for i, agent in enumerate(world.agents):
            color_slider = (i+1)*0.25 + 0.25
            if not agent.chaser:
                agent.color = np.array([0.35, 0.85, 0.35])
            else:
                agent.color = np.array([color_slider, 0.35, 1. - color_slider])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for i, agent in enumerate(world.agents):
            if world.random_init:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                x = 2. * float(i + 1) / (len(world.agents) + 1) - 1
                agent.state.p_pos = np.array([x, 0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if world.random_init:
                    landmark.state.p_pos = np.random.uniform(
                            -0.9, +0.9, world.dim_p)
                else:
                    x = 2. * float(i + 1) / (len(world.landmarks) + 1) - 1
                    landmark.state.p_pos = np.array([x, x])
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.chaser:
            collisions = 0
            for a in self.runner_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def runner_agents(self, world):
        """Return all agents that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.chaser]

    def chaser_agents(self, world):
        """Return all adversarial agents.
        """
        return [agent for agent in world.agents if agent.chaser]

    def reward(self, agent, world):
        """Agents are rewarded based on minimum agent distance to each landmark.
        """
        main_reward = (self.chaser_reward(agent, world) if agent.chaser
                       else self.runner_reward(agent, world))
        return main_reward

    def runner_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0.0
        shape = False
        chasers = self.chaser_agents(world)
        if shape:
            # reward can optionally be shaped
            # (increased reward for increased distance from adversary)
            for adv in chasers:
                rew += 0.1 * np.sqrt(np.sum(np.square(
                    agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in chasers:
                if self.is_collision(a, agent):
                    rew -= 10.

        # agents are penalized for exiting the screen,
        # so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 100)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def chaser_reward(self, agent, world):

        # Chasers are rewarded for collisions with agents
        rew = 0.0
        shape = False
        runners = self.runner_agents(world)
        chasers = self.chaser_agents(world)
        if shape:
            # reward can optionally be shaped
            # (decreased reward for increased distance from agents)
            for adv in chasers:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(
                    a.state.p_pos - adv.state.p_pos))) for a in runners])
        if agent.collide:
            for ag in runners:
                for adv in chasers:
                    if self.is_collision(ag, adv):
                        rew += 10.

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_diff = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos_diff.append(entity.state.p_pos - agent.state.p_pos)
                # entity_pos.append(entity.state.p_pos)

        # communication of all other agents
        other_pos = []
        other_vel = []
        other_pos_diff = []
        other_vel_diff = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos)
            other_vel.append(other.state.p_vel)
            other_pos_diff.append(other.state.p_pos - agent.state.p_pos)
            other_vel_diff.append(other.state.p_vel - agent.state.p_vel)

        obs = np.concatenate(
                # [agent.state.p_pos] +
                # [agent.state.p_vel] +
                entity_pos_diff +
                # other_pos +
                other_pos_diff +
                # other_vel +
                other_vel_diff)
        return obs

    def done(self, agent, world):
        return world.num_steps >= world.max_steps
