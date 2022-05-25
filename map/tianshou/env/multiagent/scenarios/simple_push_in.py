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
        num_good_agents = self.params['num_good_agents'] # 1
        num_adversaries = self.params['num_adversaries'] # 1
        assert (num_good_agents > 0), 'invalid num of good agents'
        assert (num_adversaries > 0), 'invalid num of adversaries'
        assert (num_good_agents == num_adversaries), 'num of adversaries should be equal to good agents'
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_good_agents
        self.world_radius = 1
        world.obs_radius = self.params['obs_radius']
        world.rew_shaping = self.params['rew_shape']
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        self.circ_radius = self.world_radius - max([landmark.size for landmark in world.landmarks] + [a.size for a in world.agents if a.adversary])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.1, 0.1, 0.1])
            # landmark.color[(i + 1) % 3] += 0.8
            if i == 0:
                landmark.color = np.array([0.1, 0.9, 0.1])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.index = i
        # set goal landmark
        # goal = np.random.choice(world.landmarks)
        goal = world.landmarks[0]
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                # agent.color = np.array([0.75, 0.25, 0.25])
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                j = goal.index
                # agent.color[(j + 1) % 3] += 0.5
                agent.color = np.array([0.85, 0.35, 0.35])

        # set random initial states
        # adversaries and landmarks are all on the circle
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
            if self.params['amb_init']:
                landmark.state.p_pos = np.zeros(world.dim_p)
                landmark.state.p_pos[0] = np.random.uniform(-self.circ_radius, self.circ_radius, 1)
                landmark.state.p_pos[1] = np.sqrt(self.circ_radius ** 2 - landmark.state.p_pos[0] ** 2)
                landmark.state.p_pos[1] *= (1 if np.random.uniform(0, 1, 1) > 0.5 else -1)
            else:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def distance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min
    
    # def benchmark_data(self, agent, world):
    #     # returns data for benchmarking purposes
    #     occupied = 0
    #     if self.is_collision(agent, agent.goal_a):
    #         occupied += 1
    #     if agent.adversary:
    #         # min adv-goal distance
    #         dists = [self.distance(a.state_p_pos, a.goal_a.state.p_pos) for a in self.adversaries(world)]
    #         rew = self.adversary_reward(agent, world)
    #         return rew, occupied, min(dists) / len(self.adversaries(world))
    #     else:
    #         # min agent-goal distance
    #         dists = [self.distance(a.state_p_pos, a.goal_a.state.p_pos) for a in self.good_agents(world)]
    #         rew = self.agent_reward(agent, world)
    #         return rew, occupied, min(dists) / len(self.good_agents(world))

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        occupied = 0
        pos_rew, adv_rew = 0.0, 0.0
        bound_rew = 0.0

        ag_dist = self.distance(agent.state.p_pos, agent.goal_a.state.p_pos)
        if ag_dist < agent.size + agent.goal_a.size:
            occupied += 1
        if agent.adversary:
            rew = self.adversary_reward(agent, world)
        else:
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    adv_rew -= 5.0
            if self.is_collision(agent, agent.goal_a):
                pos_rew += 5.0
            rew = pos_rew + adv_rew
            def bound(x):
                if x > 1.0:
                    return min(np.exp(2 * x - 2), 10)
                else:
                    return 0.0
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                bound_rew -= bound(x)
            rew += bound_rew
            # rew = self.agent_reward(agent, world)
        return rew, occupied, ag_dist, pos_rew, bound_rew
        

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            return self.adversary_reward(agent, world)
        else:
            return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        if world.rew_shaping:
            return -np.sqrt(np.sum(np.square(
            agent.state.p_pos - agent.goal_a.state.p_pos)))
        else:
            pos_rew, adv_rew = 0.0, 0.0
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    adv_rew -= 5.0
            if self.is_collision(agent, agent.goal_a):
                pos_rew += 5.0
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
        if world.rew_shaping:
            # keep the nearest good agents away from the goal
            agent_dist = [np.sqrt(np.sum(np.square(
                a.state.p_pos - a.goal_a.state.p_pos)))
                for a in world.agents if not a.adversary]
            pos_rew = min(agent_dist)
            neg_rew = np.sqrt(
                np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
            return pos_rew - neg_rew
        else:
            rew = 0.0
            for a in self.good_agents(world):
                if self.is_collision(a, a.goal_a):
                    rew -= 5.0
                # if self.is_collision(a, agent):
                #     rew += 5.0      
            if self.is_collision(agent, agent.goal_a):
                rew += 5.0
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
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

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        if not agent.adversary:
            return np.concatenate(
                [agent_vis_mask] +
                [lm_vis_mask] +
                agent_pos +
                agent_vel +
                entity_pos +
                [agent.goal_a.state.p_pos] +
                [agent.color] +
                entity_color
            )
        else:
            # other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate(
                [agent_vis_mask] +
                [lm_vis_mask] + 
                agent_pos +
                agent_vel +
                entity_pos +
                [np.zeros(5 + len(world.landmarks) * 3)])

    def done(self, agent, world):
        return world.num_steps >= world.max_steps
