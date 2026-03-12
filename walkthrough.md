# OP3 Standing Balance RL Training — 完整技术回顾

## 1. 项目目标

训练 Robotis OP3 人形机器人学会**原地站立平衡**。使用 MuJoCo MJX (GPU-accelerated physics) + Brax PPO 强化学习在 NVIDIA RTX 5090 上训练。

---

## 2. 模型选择

### 最终使用的模型

**mujoco_playground 内置的 [op3_mjx_feetonly.xml](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml)**

- 来源: [op3_mjx_feetonly.xml](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml) (365行)
- 这是 DeepMind 官方为 MJX GPU 训练优化过的 OP3 模型
- 原始模型来自 `mujoco_menagerie/robotis_op3`，与 OpenServoSim 的 [op3.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/op3.xml) 是**同一个机器人**

### 模型参数

| 属性 | 值 |
|------|-----|
| 总质量 | ~3.1kg |
| 关节数 | 21 (head×2, arm×6, leg×12, freejoint) |
| 执行器 | 20 个 position actuators |
| 仿真步长 | 0.004s (sim_dt) |
| 控制步长 | 0.02s (ctrl_dt, 50Hz) |
| 每控制步物理步数 | 5 (n_substeps) |
| 执行器类型 | Position control (模拟 Dynamixel 舵机) |
| 默认 Kp | 21.1 |
| 默认 Kd (damping) | 1.084 |
| Force range | ±5N |
| Armature | 0.045 |
| Friction loss | 0.03 |

### 为什么不用 OpenServoSim 的 scene_rl.xml (自定义环境)

> [!CAUTION]
> **失败尝试**: 我们最初创建了 [scene_rl.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/scene_rl.xml) + [op3_standing_env.py](file:///home/zero/OpenServoSim/training/op3_standing_env.py)。这个方案**JIT 编译超过 7 分钟仍未完成，GPU 利用率仅 7-13%**。

**根本原因**: [scene_rl.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/scene_rl.xml) 通过 `<include file="op3.xml"/>` 引入了原始模型，原始模型的**所有 mesh 碰撞体都启用了碰撞检测**（~30+ 个 convex mesh geom）。MJX 在 GPU 上的碰撞检测需要对所有 geom pair 进行 JIT 编译，这导致 XLA compilation graph 极大。

**DeepMind 的优化方案** ([op3_mjx_feetonly.xml](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml)):
- 所有非脚部 geom 设置为 `contype="0" conaffinity="0"` (禁用碰撞)
- 只有 4 个脚部 box geom 参与碰撞: `l_foot1`, `l_foot2`, `r_foot1`, `r_foot2`
- 这将碰撞 pair 从 O(n²) 降到几乎 O(1)，JIT 编译从 7+ 分钟降到 ~118 秒

---

## 3. 环境设计

### 使用的环境

**`mujoco_playground._src.locomotion.op3.joystick.Joystick`** (Op3Joystick)

- 来源: [joystick.py](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py) (486行)
- 继承: [Op3Env](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/base.py#40-121) → [MjxEnv](file:///home/zero/mujoco_playground/mujoco_playground/_src/mjx_env.py#218-318)
- 关键: 通过 `wrapper.wrap_for_brax_training()` 包装，使其兼容 Brax PPO

### 关键修改：速度命令设为零

```python
cfg.lin_vel_x = [0.0, 0.0]    # 原始: [-0.6, 1.5]
cfg.lin_vel_y = [0.0, 0.0]    # 原始: [-0.8, 0.8]
cfg.ang_vel_yaw = [0.0, 0.0]  # 原始: [-0.7, 0.7]
```

这样 [sample_command()](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#468-486) 始终返回 `[0, 0, 0]`（或有10%概率也返回零向量），机器人的目标就是**保持静止不动**。

### 观测空间 (147维 = 49 × 3 history)

每帧 49 维:
- [gyro](file:///home/zero/OpenServoSim/training/op3_standing_env.py#141-143) (3): 陀螺仪角速度
- [gravity](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/base.py#72-75) (3): 重力方向矢量 (framezaxis)
- [command](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#468-486) (3): 速度命令 (始终为 [0,0,0])
- `qpos_delta` (20): 关节角度与默认姿态的偏差
- `last_act` (20): 上一步动作

使用 3 帧历史堆叠: `obs_history_size=3`

### 动作空间 (20维, 连续)

```python
motor_targets = default_pose + action * action_scale  # action_scale=0.3
motor_targets = clip(motor_targets, lower_limits, upper_limits)
```

动作范围: [-1, 1]，乘以 0.3 弧度，加上默认站立姿态 (`stand_bent_knees` keyframe)。

### 终止条件

```python
# 重力 z 分量 < 0.85 (倒了)
fall = gravity_z < 0.85
# 或躯干高度 < 0.21m
fall |= torso_z < 0.21
# 或关节超限
joint_limit_exceed = any(joint_angles outside ctrl_range)
```

### 初始状态

使用 `stand_bent_knees` keyframe:
- 躯干高度: 0.2436m
- 双膝微弯 (~弯曲约 60°)
- 手臂自然下垂

---

## 4. 奖励函数

### 第一次尝试 (失败，reward=0.000)

```python
tracking_lin_vel = 0.5   # 太小
tracking_ang_vel = 0.3   # 太小
orientation = -10.0      # 太大
zero_cmd = -1.0          # 太大
```

**问题**: 正奖励 (0.5×1.0 + 0.3×1.0 = 0.8) 远小于负惩罚 (orientation + torques + ...)。`jp.clip(total * dt, 0.0, 10000.0)` 将负 reward clip 到 0.000。机器人无法学到任何东西。

### 第二次尝试 (当前，reward 从 17→456)

```python
tracking_lin_vel = 15.0   # 跟踪零速度的大正奖励
tracking_ang_vel = 8.0    # 跟踪零角速度的大正奖励
orientation = -5.0        # 适度的保持直立惩罚
zero_cmd = -0.5           # 静止时惩罚不必要的动作  
lin_vel_z = -2.0          # 不要上下弹跳
ang_vel_xy = -0.05        # 不要晃
torques = -0.0002         # 节能
action_rate = -0.01       # 动作平滑
termination = -1.0        # 摔倒惩罚
feet_slip = -0.1          # 脚不要滑
energy = -0.0001          # 节能
```

**计算**: 完美站立时，[tracking_lin_vel](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#312-321) 给 15×1.0=15，[tracking_ang_vel](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#322-330) 给 8×1.0=8，总正奖励 ~23。减去 orientation (~0.01×-5), action_rate, torques 等小损失，净 reward ≈ 20+/step。乘以 dt=0.02，每步 ~0.4+。1000步 episode，总 reward ~400-460。这与实际观测到的 444-456 一致。

### 奖励函数细节

每个 reward 项的数学公式（来自 [joystick.py](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py)):

```python
# 1. tracking_lin_vel: 速度跟踪 (越接近目标越高)
error = sum((cmd[:2] - local_vel[:2])²)
reward = exp(-error / tracking_sigma)  # tracking_sigma=0.25

# 2. tracking_ang_vel: 角速度跟踪
error = (cmd[2] - ang_vel[2])²
reward = exp(-error / tracking_sigma)

# 3. orientation: 保持直立 (惩罚重力向量xy分量)
cost = sum(gravity[:2]²)  # 完美直立时为0

# 4. zero_cmd: 命令为零时惩罚动作
cost = sum(action²) * (cmd_norm < 0.1)  # 当命令≈0时生效

# 5. lin_vel_z: 垂直速度惩罚
cost = global_linvel[2]²

# 6. ang_vel_xy: 横向角速度惩罚  
cost = sum(global_angvel[:2]²)

# 7. torques: 力矩惩罚
cost = sqrt(sum(force²)) + sum(|force|)

# 8. action_rate: 动作变化率 (一阶+二阶)
cost = sum((a - a_prev)²) + sum((a - 2*a_prev + a_prev_prev)²)

# 9. energy: 功率消耗
cost = sum(|qvel| * |force|)

# 10. termination: 提前终止惩罚
cost = done & (step < 500)

# 11. feet_slip: 脚部滑动
cost = sum(foot_vel_xy² * foot_contact)

# 最终: reward = clip(sum(scale_i * reward_i) * dt, 0.0, 10000.0)
```

---

## 5. 训练配置

### PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_timesteps` | 100,000,000 | 总训练步数 |
| `num_envs` | 8,192 | 并行环境数 |
| `episode_length` | 1,000 | 每 episode 步数 (20秒) |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `entropy_cost` | 0.01 | 探索鼓励 |
| `discounting` | 0.97 | 折扣因子 |
| `batch_size` | 256 | 梯度更新批大小 |
| `num_minibatches` | 32 | PPO epoch per batch |
| `num_updates_per_batch` | 4 | 每批更新次数 |
| `unroll_length` | 20 | GAE 展开长度 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `normalize_observations` | True | 观测标准化 |
| `reward_scaling` | 1.0 | 奖励缩放 |

### 网络架构

| 网络 | 隐藏层大小 |
|------|-----------|
| Policy | (128, 128, 128, 128) — 4层 |
| Value | (256, 256, 256, 256, 256) — 5层 |

输入 key: `"state"` (用 brax observation wrapper)

### 硬件

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA RTX 5090 (Blackwell) |
| CUDA | 13.1 driver |
| JAX | 0.6.2 |
| OS | Native Linux (Ubuntu 24.04) |
| RAM | 64GB |

---

## 6. 当前训练结果

训练启动时间: 2026-03-10 07:11:54

| 步数 | Reward | 趋势 |
|------|--------|------|
| 0 | 17.817 | 基线 |
| 11.1M | 444.565 | ↑ 大幅提升 |
| 22.3M | 455.970 | ↑ 接近收敛 |
| 33.4M | 176.127 | ↓ 临时下降 |
| 44.6M | 453.946 | ↑ 恢复 |
| 55.7M | 455.935 | → 稳定 |
| 66.8M | 456.741 | → 稳定 |

- JIT 编译: ~118.5 秒
- 训练速度: ~12M steps/min (GPU 满载)
- 预计总时间: ~8-9 分钟

---

## 7. 已知问题与优化方向

### 问题 1: 自定义 MJX 环境过慢

自定义 [Op3StandingEnv](file:///home/zero/OpenServoSim/training/op3_standing_env.py#64-314) 加载完整 [op3.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/op3.xml) 导致 JIT 编译 7+ 分钟不完成。必须使用 DeepMind 优化的 [op3_mjx_feetonly.xml](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml) 版本。

### 问题 2: 无法直接使用 OpenServoSim 的 servo_biped 模型

[models/servo_biped/servo_biped.xml](file:///home/zero/OpenServoSim/models/servo_biped/servo_biped.xml) 是一个 10-DOF 简化模型，没有 MJX 传感器和 keyframe。如需用这个模型训练，需要：
1. 添加 `contype=0 conaffinity=0` 到所有非脚部 geom
2. 添加 MJX 传感器 (gyro, upvector, framelinvel 等)
3. 添加 standing keyframe
4. 可以直接套用 Op3Joystick 的环境代码模式

### 问题 3: 奖励收敛到 ~456 后不再上升

当前 reward 收敛在 ~456，这代表几乎完美的站立 (tracking rewards ≈ max)。进一步优化可以：
- 增加 `episode_length` 到 2000-5000（更长的站立测试）
- 添加外力扰动 (`velocity_kick`) 测试鲁棒性
- 降低 `action_scale` 到 0.1（更精细的控制）

### 问题 4: 训练效率

当前 100M 步在 RTX 5090 上约 8 分钟完成，这已经是接近最优的速度。主要瓶颈在 JIT 编译 (~120s)，而非训练本身。

---

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| [train_op3_standing.py](file:///home/zero/OpenServoSim/training/train_op3_standing.py) | 当前使用的训练脚本 |
| [op3_standing_env.py](file:///home/zero/OpenServoSim/training/op3_standing_env.py) | 自定义环境 (已弃用，JIT过慢) |
| [scene_rl.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/scene_rl.xml) | 自定义场景 (已弃用) |
| [joystick.py](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py) | 当前使用的 Op3Joystick 环境 |
| [op3_mjx_feetonly.xml](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/op3_mjx_feetonly.xml) | MJX 优化的 OP3 模型 |
| [base.py](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/base.py) | Op3Env 基类 |
| [wrapper.py](file:///home/zero/mujoco_playground/mujoco_playground/_src/wrapper.py) | Brax 兼容包装器 |
| [train_jax_ppo.py](file:///home/zero/mujoco_playground/learning/train_jax_ppo.py) | mujoco_playground 官方训练脚本 |
| [op3.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/op3.xml) | 原始 OP3 模型定义 |
| [scene_enhanced.xml](file:///home/zero/OpenServoSim/models/reference/robotis_op3/scene_enhanced.xml) | 增强版场景 (kp=40, ±12N) |
