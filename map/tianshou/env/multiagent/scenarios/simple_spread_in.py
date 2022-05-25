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

        # set any world properties first
        world.dim_c = 2
        num_agents = self.params['num_good_agents']
        num_landmarks = num_agents
        world.collaborative = True
        world.rew_shaping = self.params['rew_shape']
        self.world_radius = 1.0
        world.obs_radius = self.params['obs_radius']
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        self.circ_radius = self.world_radius - max([landmark.size for landmark in world.landmarks])
        # add agents
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0
        self.end_steps = world.max_steps

        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = np.array([0.85, 0.35, 0.35])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = np.array([0.1, 0.9, 0.1])

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
                landmark.state.p_pos[0] = np.random.uniform(-self.circ_radius, self.circ_radius, 1)
                landmark.state.p_pos[1] = np.sqrt(self.circ_radius ** 2 - landmark.state.p_pos[0] ** 2)
                landmark.state.p_pos[1] *= (1 if np.random.uniform(0, 1, 1) > 0.5 else -1)
            else:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # rew = self.reward(agent, world)
        rew = 0.0
        dists = []
        for land in world.landmarks:
            al_dist = self.distance(agent.state.p_pos, land.state.p_pos)
            dists.append(al_dist)
        min_dist = min(dists) 
        occupied_landmarks = 1 if min_dist < agent.size + land.size else 0
        end_steps = self.end_steps

        win_agents = []
        for land in world.landmarks:
            for a in world.agents:
                if self.is_collision(a, land):
                    win_agents.append(a)
                    break
        rew += 2 * len(set(win_agents))

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
        return (rew, min_dist, occupied_landmarks, end_steps, rew-bound_rew, bound_rew)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance
    #     # to each landmark, penalized for collisions
    #     rew = 0
    #     for land in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(
    #             a.state.p_pos - land.state.p_pos))) for a in world.agents]
    #         rew -= min(dists)
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #     return rew

    def reward(self, agent, world):
        # Agents are rewarded based on if they reached/occupied a landmark. 
        # Occupancy can only be one-to-one.
        rew = 0.0
        if world.rew_shaping:
            for land in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(
                    a.state.p_pos - land.state.p_pos))) for a in world.agents]
                rew -= min(dists)
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
            return [rew]
        else:
            win_agents = []
            for land in world.landmarks:
                for a in world.agents:
                    if self.is_collision(a, land):
                        win_agents.append(a)
                        break
            rew += 2 * len(set(win_agents))

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
        win_agents = []
        for land in world.landmarks:
            for a in world.agents:
                if self.is_collision(a, land):
                    win_agents.append(a)
                    break
        if len(set(win_agents)) == len(world.agents):
            if world.num_steps < self.end_steps:
                self.end_steps = world.num_steps 
            return True
        else:
            return world.num_steps >= world.max_steps
