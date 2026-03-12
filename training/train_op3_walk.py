"""
OpenServoSim - Train OP3 Fast Walking Policy

Uses Op3Joystick environment with high forward velocity commands and
reward weights optimized for fast, stable bipedal walking.

Key design:
  - High forward velocity target: [1.0, 2.0] m/s
  - Strong orientation penalty: robot must stay upright (no crawling)
  - Strong height bonus: must maintain standing height
  - Boosted tracking reward: encourage matching velocity commands
  - Velocity kick perturbations: build robustness

Usage:
  cd /home/zero/mujoco_playground && source .venv/bin/activate
  python /home/zero/OpenServoSim/training/train_op3_walk.py
"""

import os
import functools
import time
import json
import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src.locomotion.op3 import joystick as op3_joystick
from mujoco_playground import wrapper


# ===== Configuration =====
NUM_TIMESTEPS = 200_000_000   # 200M steps for walking (harder task)
NUM_ENVS = 8192
NUM_EVALS = 10
EPISODE_LENGTH = 1000
SEED = 42

PPO_PARAMS = dict(
    num_timesteps=NUM_TIMESTEPS,
    num_evals=NUM_EVALS,
    reward_scaling=1.0,
    episode_length=EPISODE_LENGTH,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=NUM_ENVS,
    batch_size=256,
    max_grad_norm=1.0,
)

NETWORK_FACTORY = dict(
    policy_hidden_layer_sizes=(128, 128, 128, 128),
    value_hidden_layer_sizes=(256, 256, 256, 256, 256),
    policy_obs_key="state",
    value_obs_key="state",
)


def main():
    print("=" * 70)
    print("  OpenServoSim - OP3 Fast Walking RL Training")
    print("=" * 70)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"  JAX backend: {backend}")
    print(f"  Devices: {devices}")

    # Configure Op3Joystick for fast walking
    cfg = op3_joystick.default_config()

    # High forward velocity, moderate lateral and turning
    cfg.lin_vel_x = [0.5, 2.0]      # Fast forward walking (0.5-2.0 m/s)
    cfg.lin_vel_y = [-0.3, 0.3]     # Small lateral motion
    cfg.ang_vel_yaw = [-0.5, 0.5]   # Moderate turning

    # Reward weights optimized for fast, stable bipedal walking
    cfg.reward_config.scales.tracking_lin_vel = 10.0   # Strong incentive to match velocity
    cfg.reward_config.scales.tracking_ang_vel = 5.0    # Match turning commands
    cfg.reward_config.scales.orientation = -8.0        # MUST stay upright (prevents crawling)
    cfg.reward_config.scales.lin_vel_z = -2.0          # No bouncing
    cfg.reward_config.scales.ang_vel_xy = -0.1         # No wobbling
    cfg.reward_config.scales.torques = -0.0001         # Slight energy efficiency
    cfg.reward_config.scales.action_rate = -0.01       # Smooth motion
    cfg.reward_config.scales.zero_cmd = -0.5           # Quiet when stopped
    cfg.reward_config.scales.termination = -1.0        # Don't fall
    cfg.reward_config.scales.feet_slip = -0.1          # No foot sliding
    cfg.reward_config.scales.feet_clearance = 0.5      # Lift feet properly
    cfg.reward_config.scales.energy = -0.0001          # Power efficiency

    # Velocity perturbations for robustness
    cfg.velocity_kick = [1.0, 5.0]
    cfg.kick_durations = [0.05, 0.2]
    cfg.kick_wait_times = [1.0, 3.0]

    # Action scale: slightly larger for walking
    cfg.action_scale = 0.3

    print(f"\n  Loading Op3Joystick (fast walking mode)...")
    env = op3_joystick.Joystick(config=cfg)
    print(f"  Action size: {env.action_size}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Control dt: {env.dt}s, Sim dt: {env.sim_dt}s")
    print(f"  Velocity range: x={cfg.lin_vel_x}, y={cfg.lin_vel_y}, yaw={cfg.ang_vel_yaw}")

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"Op3Walk-{timestamp}"
    logdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", exp_name
    )
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"  Experiment: {exp_name}")
    print(f"  Checkpoints: {ckpt_path}")

    # Save config
    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    # Network
    network_fn = functools.partial(
        ppo_networks.make_ppo_networks, **NETWORK_FACTORY
    )

    # Eval env
    eval_env = op3_joystick.Joystick(config=cfg)

    times = [time.monotonic()]

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        r = metrics.get("eval/episode_reward", 0)
        print(f"  {num_steps}: reward={float(r):.3f}")

    print(f"\n  PPO: {NUM_TIMESTEPS//1_000_000}M steps, {NUM_ENVS} envs")
    print(f"  Starting training...")
    print(f"  {'━' * 60}")

    train_fn = functools.partial(
        ppo.train,
        **PPO_PARAMS,
        network_factory=network_fn,
        seed=SEED,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=128,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    print(f"  {'━' * 60}")
    if len(times) > 1:
        print(f"  JIT compile: {times[1] - times[0]:.1f}s")
        print(f"  Training: {times[-1] - times[1]:.1f}s")
    print(f"  Training complete!")

    # Render rollout video
    print(f"\n  Rendering rollout video...")
    try:
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        infer_env = op3_joystick.Joystick(config=cfg)
        wrapped_env = wrapper.wrap_for_brax_training(
            infer_env, episode_length=EPISODE_LENGTH, action_repeat=1,
        )

        NUM_ROLLOUTS = 1
        rng = jax.random.split(jax.random.PRNGKey(SEED), NUM_ROLLOUTS)
        reset_states = jax.jit(wrapped_env.reset)(rng)

        def step_fn(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)
            act_keys = jax.random.split(act_key, NUM_ROLLOUTS)
            act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
            state = wrapped_env.step(state, act)
            return (state, rng), state

        @jax.jit
        def do_rollout(state, rng):
            _, traj = jax.lax.scan(
                step_fn, (state, rng), None, length=500
            )
            return traj

        print(f"  Running rollout (JIT compiling)...")
        traj = do_rollout(reset_states, jax.random.PRNGKey(SEED + 1))

        # Render frames
        print(f"  Rendering frames...")
        render_every = 2
        fps = 1.0 / infer_env.dt / render_every

        qposes = traj.data.qpos[::render_every, 0]
        renderer = mujoco.Renderer(infer_env.mj_model, height=480, width=640)
        mj_data = mujoco.MjData(infer_env.mj_model)
        frames = []
        for i in range(qposes.shape[0]):
            mj_data.qpos[:] = np.array(qposes[i])
            mujoco.mj_forward(infer_env.mj_model, mj_data)
            renderer.update_scene(mj_data, camera="track")
            frames.append(renderer.render())
        renderer.close()

        import mediapy as media
        video_path = os.path.join(logdir, "rollout.mp4")
        media.write_video(video_path, frames, fps=fps)
        print(f"  Video saved: {video_path}")
    except Exception as e:
        print(f"  Video rendering failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  All done! Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
