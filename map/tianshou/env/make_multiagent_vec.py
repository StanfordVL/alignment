def make_env_vec(scenario_name, benchmark=False, optional=None):
    from tianshou.env.multiagent.environment_vec import MultiAgentEnv
    import tianshou.env.multiagent.scenarios as scenarios

    # load scenario from script
    if optional is None:
        scenario = scenarios.load(scenario_name + ".py").Scenario()
    else:
        scenario = scenarios.load(scenario_name + ".py").Scenario(**optional)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.done, 
                            cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.done,
                            cam_range=scenario.world_radius)
    return env