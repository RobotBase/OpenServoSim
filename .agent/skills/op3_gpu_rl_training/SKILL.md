---
description: How to train OP3 walking policy with GPU-accelerated PPO using mujoco_playground on native Linux (RTX 5090)
---

# OP3 GPU RL Training — Native Linux (No WSL2 Required)

This skill documents the **verified, working** workflow for training RL locomotion policies for the OP3 robot using mujoco_playground's built-in PPO trainer on native Linux with GPU acceleration.

> **Supersedes** the earlier `mujoco_playground_rl_training` skill which documented a WSL2-based CPU-only workflow. As of 2026-03-10, JAX 0.6.2 with CUDA works natively on RTX 5090.

---

## Prerequisites

| Component | Location / Version |
|-----------|--------------------|
| mujoco_playground repo | `/home/zero/mujoco_playground` |
| Python venv | `/home/zero/mujoco_playground/.venv` |
| Package manager | `uv` |
| JAX | 0.6.2 (GPU backend, CudaDevice) |
| GPU | RTX 5090 (Blackwell), CUDA 13.1 driver |
| OS | Native Linux (not WSL2) |

### Sync dependencies

```bash
cd /home/zero/mujoco_playground
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
uv --no-config sync --all-extras
```

### Verify JAX GPU

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
# Expected: gpu [CudaDevice(id=0)]
```

---

## Training

Use mujoco_playground's official training script directly. It handles the Brax wrapper compatibility internally via `wrapper.wrap_for_brax_training`.

### Quick start (100M steps, ~7.5 min on RTX 5090)

```bash
cd /home/zero/mujoco_playground
source .venv/bin/activate

python learning/train_jax_ppo.py \
  --env_name=Op3Joystick \
  --num_timesteps=100000000 \
  --num_envs=8192 \
  --num_evals=10 \
  --seed=42
```

### What happens

1. **First run only**: auto-clones `mujoco_menagerie` (~1-2 min)
2. **JIT compilation**: ~116s (XLA autotuning warnings for Blackwell are normal)
3. **Training**: ~450s for 100M steps
4. **Video rendering**: crashes on `vision_config` for Op3 (non-critical, checkpoint is saved)

### Key PPO parameters (auto-loaded from `locomotion_params`)

| Parameter | Value |
|-----------|-------|
| Policy network | 4×128 hidden |
| Value network | 5×256 hidden |
| Learning rate | 3e-4 |
| Discounting | 0.97 |
| Entropy cost | 0.01 |
| Num minibatches | 32 |
| Unroll length | 20 |
| Batch size | 256 |
| Reward scaling | 1.0 |

### Output

- **Logs**: `logs/Op3Joystick-<TIMESTAMP>/`
- **Checkpoints**: `logs/Op3Joystick-<TIMESTAMP>/checkpoints/<STEP_NUM>/` (orbax format)
- **Config**: `logs/Op3Joystick-<TIMESTAMP>/checkpoints/config.json`

---

## Verified Training Results (2026-03-10)

| Metric | Value |
|--------|-------|
| Total steps | 100,000,000 |
| Parallel envs | 8,192 |
| JIT compile time | 116s |
| Training time | 447s (7.5 min) |
| Final reward | 17.832 |
| Peak reward | 19.077 (@ 57M steps) |

### Reward curve

```
  Steps (M)   Reward
  ─────────   ──────
   0.0         0.158
  11.5        11.634
  22.9        16.096
  34.4        16.993
  45.9        16.377
  57.3        19.077  ← peak
  68.8        16.975
  80.3        17.401
  91.8        17.616
 103.2        17.832
```

Checkpoint: `/home/zero/mujoco_playground/logs/Op3Joystick-20260310-055808/checkpoints/000045875200`

---

## Rendering Rollout Videos

The built-in video rendering in `train_jax_ppo.py` crashes for Op3 (missing `vision_config`). Use this standalone script instead:

```bash
cd /home/zero/mujoco_playground
source .venv/bin/activate
python /tmp/render_op3_rollout.py
```

Key points for the render script:
- Set `CKPT_DIR` to the **numbered** checkpoint directory, e.g. `checkpoints/000045875200`
- Uses `ppo.train()` with `num_timesteps=0` + `restore_checkpoint_path` to load params
- Renders via `infer_env.render()` + `mediapy.write_video()`
- Output: `rollout.mp4` (500 frames, 25fps, 20 seconds)

---

## Known Issues

### 1. Post-training video crash
`train_jax_ppo.py` line 478 accesses `env_cfg.vision_config.nworld` which doesn't exist for Op3Joystick. Training and checkpoint saving complete before this point — only the automatic video rendering fails.

### 2. XLA autotuning warnings
Blackwell GPUs produce many `dot_search_space` warnings during JIT. These are harmless — XLA falls back to the full hints set.

### 3. Brax wrapper compatibility (SOLVED)
The old `mujoco_playground_rl_training` skill documented that `State.pipeline_state` was missing. This is now handled by `mujoco_playground._src.wrapper.wrap_for_brax_training()` which wraps the environment with `VmapWrapper`, `EpisodeWrapper`, and `BraxAutoResetWrapper`. **No custom wrapper needed.**

### 4. Old CUDA 12 segfault (RESOLVED)
The old skill documented JAX CUDA 12 segfaulting on RTX 5090. With JAX 0.6.2, CUDA works natively. No plugin uninstall needed.

---

## Next Steps

1. **Tune reward**: Adjust reward weights in `Op3Joystick` config via `--playground_config_overrides`
2. **Longer training**: Try 200M-500M steps for higher asymptotic reward
3. **Domain randomization**: Add `--domain_randomization` for sim-to-real robustness
4. **Inference deployment**: Port checkpoint to `OpenServoSim/examples/04_rl_inference.py` for MuJoCo viewer with WASD control
5. **Sim-to-real**: Map policy outputs to real Dynamixel servo commands via OpenServoSim HAL

---

## File References

| File | Purpose |
|------|---------|
| `/home/zero/mujoco_playground/learning/train_jax_ppo.py` | Official PPO training script |
| `/home/zero/mujoco_playground/mujoco_playground/_src/wrapper.py` | Brax compatibility wrappers |
| `/home/zero/mujoco_playground/mujoco_playground/config/locomotion_params.py` | Default PPO hyperparams per env |
| `/home/zero/OpenServoSim/training/train_op3_walk.py` | Old custom training script (superseded) |
| `/home/zero/OpenServoSim/examples/04_rl_inference.py` | Inference viewer with WASD controls |
| `/home/zero/OpenServoSim/.agent/skills/mujoco_playground_rl_training/SKILL.md` | Old skill (WSL2, CPU-only) |
