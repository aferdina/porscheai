import numpy as np
import pytest

from porscheai.environment.base_env import SimpleDriver


@pytest.mark.parametrize(
    "env",
    [
        pytest.lazy_fixture("easy_general_game_config"),
    ],
)
def test_easy_general_game_config(env: SimpleDriver):
    assert env.total_no_timesteps == 1000
    assert env.time_step_size_s == 0.01
    assert env.start_velocity_ms == 0.0
    assert env.game_physics_params.velocity_ms == 0.0
    assert env.game_physics_params.current_time_step == 0
    assert env.traj_configs.total_timesteps == 1000
    assert env.traj_configs.simulation_frequency_s == 0.01
    assert np.array_equal(
        env.traj_configs.seconds_markers_s,
        np.array([0.0, 2.0, 4.0, 5.0, 7.0, 10.0], dtype=np.float32),
    )
    assert np.array_equal(
        env.traj_configs.velocities_kmh,
        np.array([0.0, 10.0, 20.0, 15.0, 20.0, 15.0], dtype=np.float32),
    )
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, truncated, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert isinstance(truncated, bool)
