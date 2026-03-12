# OP3 进阶训练计划：从步行到奔跑 (Running Policy)

## 1. 阶段回顾：我们是如何实现稳定步行的

在前面的训练中，我们成功实现了基础的“原地站立”和“快速步行”。
在步行训练中（200M steps，11.4分钟），我们摸索出了几个**关键特征和重要参数**：

### 成功的关键技术（保留与扩展）
1. **MJx 碰撞优化**：仅对脚底(feet)开启碰撞检测，避免了自碰撞带来的极大惩罚和算力浪费（JIT 仅 116s），这是我们能高吞吐量训练的基石。
2. **直立姿态强制约束 (`orientation = -8.0`)**：这是防止算法“作弊”（例如为了追求前向速度而弯腰变成爬行或翻滚）的核心参数。
3. **高跟踪奖励 (`tracking_lin_vel = 10.0`)**：让速度指令在总 reward 中占据主导，使得机器人愿意承担摔倒的风险去提速。
4. **抬腿奖励 (`feet_clearance = 0.5`)**：打破了早期拖着脚走的“滑行”步态，促成了真实的双足交替迈步。
5. **抗扰动训练 (`velocity_kick`)**：使模型对自身重心的微小偏差不再敏感。

### 限制速度的瓶颈（需要改变的参数）
1. **连续惩罚太重**：为了求稳，我们设置了 `action_rate = -0.01` (平滑度) 和 `lin_vel_z = -2.0` (不准上下弹跳)。但**跑步的本质是包含腾空期（Flight Phase）的弹跳运动**，过度惩罚 z 轴速度和高频动作，会把机器人锁死在“双脚始终贴地”的稳态步行中。
2. **缺乏步态对称性约束**：机器人在追求极限速度时，可能会发展出“一瘸一拐”或不对称的马步，这在低速下不明显，在高速下会直接导致侧翻。

---

## 2. 全新目标：设计 2x-5x 速度的奔跑模型

要让机器人跑起来（目标速度 `[2.0, 4.0] m/s` 或更高），我们需要完全重新构建训练管线。跑步与步行的本质区别在于：**跑步有双足同时离地的腾空期，重心起伏极大**。

### 策略 1：释放运动自由度，重构奖励函数 (Reward Reshaping)
我们需要解开之前绑在机器人身上的枷锁，并引入新的动态指标：
- **大幅降低或移除 [lin_vel_z](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#331-334) (防弹跳) 惩罚**：允许躯干在 z 轴上有大幅度的周期性起伏。
- **引入腾空奖励 (Flight Phase Reward)**：当两只脚同时离地且向前运动时，给予额外奖励。
- **放宽 [action_rate](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#353-360) (动作变化率)**：跑步的步频极高，需要允许甚至鼓励电机的高频、大角度输出，但同时要加大 [torques](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#343-346) 或 [energy](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#347-352) 惩罚，防止电机输出无意义的震荡震颤。
- **步态对称性奖励 (Symmetry Bonus)**：惩罚左右腿动作的相位差不是 180° 的行为，强制训练出对称的奔跑步态。

### 策略 2：课程学习 (Curriculum Learning)
直接让模型在 `3.0 m/s` 下训练几乎必然失败。我们将采用两阶段训练（或者自动课程学习）：
- **阶段一 (0-100M步)**：设定目标速度为 `[1.0, 2.5] m/s`，着重培养稳定的小跑 (Jogging) 步态。
- **阶段二 (100M-300M步)**：**加载阶段一的模型作为预训练权重**，逐步拉高目标速度至 `[2.5, 5.0] m/s`，同时加大速度跟踪权重 [tracking_lin_vel](file:///home/zero/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py#312-321)。

### 策略 3：提高物理执行极限
OP3 的电机能力是有上限的，我们需要在 MJX 配置中：
- 适当增加 `action_scale` (比如从 `0.3` 提高到 `0.5` 或 `0.7`)，让策略网络能一次性给出更激进的关节偏置。
- 可以考虑稍微调整 `Kp` (刚度)，使得腿部能在落地瞬间提供更大的反弹力（充当弹簧效应）。

---

## 3. 具体实施计划 (Implementation Pipeline)

### 第一步：编写专属的 Running Environment (`op3_running_env.py`)
我们将不能完全依赖现成的 `Op3Joystick` 配置，需要子类化或重新编写奖励逻辑：
- 增加 `air_time` (滞空时间) 奖励。
- 增加 `symmetric_gait` (对称与步频) 约束。

### 第二步：编写两阶段训练脚本 (`train_op3_run.py`)
- 设置更长的时间视野：`episode_length = 2000`（让他有机会跑长距离）。
- 设置更高的命令区间：`lin_vel_x = [1.5, 4.0]`。
- 采用更密集的网络：可以考虑把 PPO 的 policy_hidden_layer_sizes 加深到 [(256, 256, 256, 256)](file:///home/zero/mujoco_playground/mujoco_playground/_src/mjx_env.py#261-265) 提供更强表达能力。
- 增加环境数量：`NUM_ENVS = 16384`（GPU VRAM 充足，利用更多并行化寻找小概率的高速稳定解）。

### 第三步：训练与迭代
在 RTX 5090 上，预计 300M 步的训练只需要约 15-20 分钟。我们会持续监控 reward 并在训练中后期检查渲染视频的真实姿态。

---

如果这个全新管线和设计思路没有问题，我将立即开始编写并实施 `op3_running_env.py` 和 `train_op3_run.py`。
