<p align="center">
  <img src="docs/assets/logo_placeholder.png" alt="OpenServoSim" width="120" />
</p>

<h1 align="center">OpenServoSim</h1>

<p align="center">
  <strong>专为总线舵机打造的双足机器人仿真与控制框架</strong><br/>
  <em>A simulation & control framework purpose-built for serial bus servo bipedal robots</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python 3.10+" /></a>
  <a href="https://mujoco.readthedocs.io/"><img src="https://img.shields.io/badge/MuJoCo-3.x-orange.svg" alt="MuJoCo 3.x" /></a>
</p>

---

## 🔍 为什么需要这个项目？

当今主流的开源双足/四足控制框架（如 MIT Mini Cheetah、Legged Gym 等）几乎都是为**无刷直驱电机 (BLDC)** 设计的——这类电机支持高频力矩透传（1kHz+）、精确的电流环控制、以及极低的通信延迟。

然而，对于广大创客、学生和研究者而言，**低成本的总线舵机**（如 LX-15D、LX-16A、飞特 STS/SCS 系列）依然是最普及的硬件方案。但舵机有着截然不同的物理特性：

| 特性 | BLDC 关节电机 | 总线舵机 |
|------|-------------|---------|
| 控制模式 | 力矩/电流透传 | **仅位置指令** |
| 通信频率 | 1kHz+ (EtherCAT/CAN) | **50-100Hz** (半双工 UART) |
| 反馈延迟 | < 1ms | **2-5ms** (读写切换) |
| 死区 | 几乎无 | **±1-2°** |
| 内部控制 | 无（外部闭环） | 自带 PID（不可调） |

**如果直接将 BLDC 框架的高频动力学算法硬套在舵机上，结果就是：机器人疯狂抖动、电机过热、结构散架。**

**OpenServoSim** 填补了这一空白。我们从底层硬件特性出发，提供了一套从 MuJoCo 仿真到真实硬件无缝部署（Sim-to-Real）的完整解决方案。

---

## 🌟 核心特性

### 🎯 舵机物理特性仿真
不同于纯粹的刚体仿真，本框架在 MuJoCo 中针对性地模拟了舵机系统的**通信延迟**、**位置伺服死区**以及**低通滤波特性**。在电脑里跑通的算法，部署到真实硬件不会"水土不服"。

### 🧠 UVC 上体垂直控制算法
致敬并重构了 Dr. Guero 的 [UVC 算法](http://ai2001.ifdef.jp/uvc/code_Eng.html)。抛弃复杂的 ZMP 预测与动力学求解，采用基于**几何投射**的隐式积分控制器：
- 实时读取 IMU 倾斜角 → 转化为支撑腿髋关节位移修正
- **积分 + 限幅 + 衰减恢复**三段式控制，具有"记忆性"平衡
- 计算量极低，可运行于 STM32 等单片机

### 📡 总线舵机深度通信优化
专门针对半双工 UART 串行舵机的通信协议封装：
- **广播同步机制**：先逐个写入期望角度（`SERVO_MOVE_TIME_WAIT_WRITE`），再发送一帧广播（`0xFE`）同步触发所有舵机
- 完美解决串口带宽低导致的多关节动作不同步问题

### 🐍 Python 优先
告别繁琐的 C++ 编译链，全面拥抱 Python + MuJoCo 生态。算法调试、数据可视化与强化学习（RL）扩展变得前所未有地简单。

---

## 📁 项目结构

```
OpenServoSim/
├── models/                      # 机器人模型
│   ├── servo_biped/             # 10-DOF 舵机双足模型
│   │   ├── servo_biped.xml      # MuJoCo MJCF 描述文件
│   │   └── meshes/              # STL/OBJ 外观与碰撞模型
│   └── reference/               # 📚 参考模型（第三方开源）
│       ├── robotis_op3/         # DeepMind OP3 Soccer (Apache 2.0)
│       ├── darwin_op/           # DARwin-OP URDF (BSD-3)
│       ├── toddlerbot/          # Stanford ToddlerBot (MIT)
│       └── poppy_humanoid/      # Poppy Humanoid URDF (GPL v3)
├── sim/                         # 仿真环境层
│   ├── mujoco_env.py            # MuJoCo 环境封装
│   └── servo_model.py           # 💡 舵机物理特性模拟（延迟/死区/滤波）
├── controllers/                 # 控制算法层（可插拔）
│   ├── base_controller.py       # 控制器基类
│   ├── uvc_controller.py        # UVC 上体垂直控制算法
│   └── inverse_kinematics.py    # 几何逆运动学求解
├── hardware/                    # 硬件抽象层 (HAL)
│   └── servo_bus.py             # 串口总线通信驱动
├── tools/                       # 实用工具
│   ├── calibrate_offsets.py     # 舵机零位校准
│   └── visualize_gait.py        # 步态轨迹可视化
├── docs/                        # 文档
│   ├── servo_vs_bldc.md         # 舵机 vs BLDC 对比
│   └── uvc_algorithm.md         # UVC 算法详解
├── main_sim.py                  # 仿真入口
├── requirements.txt             # Python 依赖
├── THIRD_PARTY_LICENSES.md      # 第三方模型许可证
└── LICENSE                      # MIT License
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/RobotBase/OpenServoSim.git
cd OpenServoSim

# 安装依赖
pip install -r requirements.txt
```

### 运行仿真 Demo

```bash
# 运行"呼吸"测试 — 机器人膝盖做正弦波运动
python main_sim.py --render

# 运行 UVC 平衡恢复仿真（开发中）
python main_sim.py --controller uvc --render
```

---

## 🎯 适用场景

- 使用 **LX-16A / LX-15D / 飞特 STS/SCS** 等串口总线舵机搭建的双足机器人
- 主控资源有限（**STM32 / 树莓派**），无法运行复杂 MPC 算法的嵌入式设备
- 希望快速验证步态、研究传统仿生学控制或初步探索强化学习的开发者

---

## 🗺️ 路线图

- [x] 仓库架构搭建
- [ ] MuJoCo 简化模型 + 基础仿真环境
- [ ] UVC 算法 Python 实现
- [ ] 舵机延迟/死区仿真层
- [ ] LX-15D 串口通信驱动
- [ ] 完整步态 CPG 生成器
- [ ] Sim-to-Real 部署验证
- [ ] 强化学习环境封装 (Gymnasium)

---

## 🙏 致谢

- **Dr. Guero** — UVC 算法原始设计者 ([ai2001.ifdef.jp](http://ai2001.ifdef.jp/))
- **Dmitry Shalkhakov** — 算法翻译与 ODE 仿真重构
- **servo_robot_learn** — 本项目算法的参考原型

---

## 📄 License

[MIT License](LICENSE) — 自由使用、修改和分发。

---

<details>
<summary><strong>🌐 English Summary</strong></summary>

**OpenServoSim** is a lightweight, Python-first simulation and control framework designed specifically for bipedal robots powered by **serial bus servos** (e.g., LX-15D, LX-16A, Feetech STS/SCS series).

Unlike mainstream frameworks built for high-frequency BLDC motors, OpenServoSim addresses the unique challenges of servo-based robots:
- **Position-only control** (no torque passthrough)
- **High communication latency** (half-duplex UART, 2-5ms per read/write)
- **Dead zones** and **built-in PID** that can't be bypassed

Key features:
- 🎯 MuJoCo simulation with realistic servo delay, dead zone, and low-pass filter modeling
- 🧠 UVC (Upper Body Vertical Control) — a geometry-based balance algorithm that runs on microcontrollers
- 📡 Broadcast-synchronized servo bus communication for multi-joint synchronization
- 🐍 Python-first architecture for rapid prototyping and RL integration

</details>
