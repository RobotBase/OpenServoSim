"""
OpenServoSim - Train OP3 Running Policy

Uses the custom Op3RunningEnv which encourages a flight phase and relaxes 
z-axis stability requirements, designed to train a high-speed running gait.

Usage:
  cd /home/zero/mujoco_playground && source .venv/bin/activate
  PYTHONPATH=/home/zero/OpenServoSim:$PYTHONPATH python /home/zero/OpenServoSim/training/train_op3_run.py
"""

import os
import sys
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
from mujoco_playground import wrapper

# Add OpenServoSim to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.op3_running_env import Op3RunningEnv, default_running_config


# ===== Configuration =====
# Running is much harder, needs more steps and larger batch size to find stable gait
NUM_TIMESTEPS = 300_000_000
NUM_ENVS = 16384
NUM_EVALS = 15
EPISODE_LENGTH = 1500
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
    batch_size=512,
    max_grad_norm=1.0,
)

NETWORK_FACTORY = dict(
    policy_hidden_layer_sizes=(256, 256, 256, 256),  # Deeper network for running
    value_hidden_layer_sizes=(256, 256, 256, 256, 256),
    policy_obs_key="state",
    value_obs_key="state",
)


def main():
    print("=" * 70)
    print("  OpenServoSim - OP3 Running RL Training")
    print("=" * 70)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"  JAX backend: {backend}")
    print(f"  Devices: {devices}")

    # Configure Op3RunningEnv
    cfg = default_running_config()
    
    # OVERRIDE with new Linear Velocity Architecture
    cfg.reward_config.scales.tracking_lin_vel = 0.0
    cfg.reward_config.scales.forward_velocity = 8.0  # Linear gradient, ~24 pts max
    cfg.reward_config.scales.tracking_ang_vel = 1.0  # Big reduction to avoid local opt
    cfg.reward_config.scales.survival = 1.0          # Stay alive
    cfg.reward_config.scales.orientation = -3.0      # Slightly relaxed upright constraint
    
    # Disable kicks while learning to run fast
    cfg.velocity_kick = [0.0, 0.0]

    print(f"\n  Loading Op3RunningEnv...")
    env = Op3RunningEnv(config=cfg)
    print(f"  Action size: {env.action_size}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Velocity range: x={cfg.lin_vel_x}, y={cfg.lin_vel_y}, yaw={cfg.ang_vel_yaw}")
    print(f"  Action scale: {cfg.action_scale}")

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"Op3Run-{timestamp}"
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
    eval_env = Op3RunningEnv(config=cfg)

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

        infer_env = Op3RunningEnv(config=cfg)
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
                step_fn, (state, rng), None, length=800
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
