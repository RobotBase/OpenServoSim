# Guge 上半身机器人 — 模型构建经验总结

> 本文档记录了从 URDF 到 MuJoCo 仿真全过程中的关键经验和踩坑记录。
> 对后续模型维护和新模型构建具有参考价值。

---

## 1. URDF → MuJoCo MJCF 转换

### ❌ 错误方法：手动提取 euler 角

**现象**：URDF 使用 `<origin rpy="3.14 -1.57 0"/>` 格式定义关节坐标变换。手动将 RPY 转为 MuJoCo 的 `euler` 属性后，机器人关节全部扭曲变形，身体部位位置完全错误。

**原因**：
- URDF RPY 使用 **extrinsic XYZ** (固定轴) 旋转
- MuJoCo `euler` 属性使用 **intrinsic** 旋转，且依赖 `eulerseq` 设置
- 两者的旋转顺序和参考系不同，不能直接复制数值

### ✅ 正确做法：使用 MuJoCo 内置 URDF 编译器

```python
# 1. 将 URDF 中的 package:// 路径替换为相对路径
# 2. 确保 mesh 文件与 URDF 在同一目录
# 3. 使用 MuJoCo 原生加载
model = mujoco.MjModel.from_xml_path("guge.urdf")
mujoco.mj_saveLastXML("output.xml", model)
# 输出的 XML 使用正确的 quaternion 表示
```

### ⚠️ Mesh 路径问题

URDF 使用 `package://guge_urdf/meshes/base_link.STL`，MuJoCo 不认识 `package://`。
解决：复制 mesh 到临时目录 → 替换路径 → 加载后设置 `<compiler meshdir="..."/>`

### 完整转换流程

```
URDF → 修复 mesh 路径 → MuJoCo 原生编译 → mj_saveLastXML
     → 程序化添加 actuator / sensor / scene / mount stand
     → 最终 scene_guge.xml
```

---

## 2. 摄像头方向 (Camera Azimuth)

### ❌ 常见误解：azimuth=180 不是正面

机器人的朝向取决于 URDF 坐标系。本模型源自 SolidWorks，机器人面朝 **+Y 方向**。

| 视角 | azimuth | 说明 |
|------|---------|------|
| **正面** (从前往后看) | **90°** | 最常用 |
| **背面** (从后往前看) | **270°** | 检查背面 |
| **右侧** | **180°** | 看机器人右手 (画面左边) |
| **左侧** | **0°** | 看机器人左手 (画面右边) |

**推荐默认相机**: `lookat=[0,0,0.95], distance=1.0, azimuth=90, elevation=-10`

**验证方法**: 生成 8 方位网格图 (步进 45°) 确认正面方向。

---

## 3. 左右手命名约定

面对机器人时：
- **URDF 的 `left_arm`** = 画面上的 **右边** (机器人自身的左手)
- **URDF 的 `right_arm`** = 画面上的 **左边** (机器人自身的右手)

这是标准机器人学约定（和面对另一个人时一样），编程时容易搞反。

---

## 4. 关节方向参考表

### ⚠️ 极易出错：左右臂方向不对称！

通过设置每个关节 = +45° 并观察 end effector 位移方向得到：

| 关节 | +角度效果 | 功能描述 |
|------|----------|----------|
| spine1 (+) | 身体向左转 (Y+) | 腰部旋转 |
| spine2 (+) | 身体向前弯 (X+, Z-) | 鞠躬/点头 |
| spine3 (+) | 身体侧倾 (Y-) | 侧向摆动 |
| **left_arm1 (+)** | **手臂抬起** (Y+, Z+) | 肩关节外展 |
| left_arm2 (+) | 手臂前伸+上举 (X+, Z+) | 肩关节前屈 |
| left_arm3 (+) | 上臂内旋 (Y-, Z-) | 上臂旋转 |
| left_arm4 (+) | 肘部弯曲 (Y+, Z+) | 肘关节屈曲 |
| left_arm5 (+) | 腕部旋转 (Y+, Z-) | 腕关节 |
| **right_arm1 (+)** | **手臂放下** (Y-, Z-) | ⚠️ **与 left_arm1 相反！** |
| right_arm2 (+) | 复杂旋转 (X-, Y+, Z-) | 肩关节 |
| right_arm3 (+) | 微旋转 | 上臂旋转 |
| right_arm4 (+) | 肘部弯曲 (X-, Y-) | 肘关节 |
| right_arm5 (+) | 腕部旋转 (Y+) | 腕关节 |

### 🔑 关键发现

**`left_arm1(+)` 是抬手，`right_arm1(+)` 是放手！** 编写动作时：
- 抬左手 → `left_arm1` 用 **正角度**
- 抬右手 → `right_arm1` 用 **负角度**

---

## 5. 物理参数调校 — 消除重力抽搐

### 问题

上半身总质量约 3kg，重心偏离关节轴线。位置伺服的 kp 不足以抵抗重力矩，导致持续的前后摆动/抽搐。这是最耗时的问题之一。

### 参数迭代过程

| 迭代 | spine kp | spine kv | damping | armature | stiffness | spine1 最大偏差 | 结果 |
|------|----------|----------|---------|----------|-----------|---------------|------|
| 1 | 50 | — | 0.5 | 0.01 | — | **55°** | ❌ 完全不可用 |
| 2 | 500 | 50 | 2.0 | 0.02 | — | **8°** | ❌ 仍然明显摆动 |
| 3 | 5000 | 200 | 5.0 | 0.05 | — | **3.3°** | ❌ 仍有可见抖动 |
| 4 | 5000 | 200 | 5.0 | 0.05 | 1,000 | **7.3°** (spine1) 0° (spine2/3) | ❌ spine1 仍在摆 |
| 5 | 5000 | 200 | 5.0 | 0.05 | 10,000 + damp=20 | **3.2°** | ❌ 还是不行 |
| 6 | 5000 | 200 | 5.0 | 0.05 | 50,000 + damp=100 | **0.16°** | ⚠️ 接近目标 |
| **7** | **5000** | **200** | **5.0** | **0.05** | **200,000 + damp=200** | **0.014°** | ✅ **完美** |

### ✅ 最终推荐参数

```xml
<!-- 脊椎关节: 被动弹簧 + 高阻尼 = 完全消除重力摆动 -->
<joint stiffness="200000" springref="0" damping="200"/>
<position kp="5000" kv="200"/>

<!-- 手臂关节 -->
<joint damping="5.0" armature="0.05"/>  <!-- 通过 default 设置 -->
<position kp="200" kv="15"/>   <!-- 肩 (arm1, arm2) -->
<position kp="150" kv="10"/>   <!-- 上臂/肘 (arm3, arm4) -->
<position kp="100" kv="8"/>    <!-- 腕 (arm5) -->
```

### 教训

1. **仅靠 position servo (kp) 无法抵消持续重力** — kp 是弹簧力，重力是恒力，会产生稳态偏差
2. **`stiffness` (关节被动弹簧) 才是关键** — 它在关节层面提供恢复力，等效于重力补偿
3. **需要极高的 stiffness 值** — 因为真实 mesh 的质量分布复杂，简化测试的参数不能直接用
4. **spine2/spine3 比 spine1 容易稳定** — spine1 承受整个上半身的力矩，需要最高参数

---

## 6. 动作设计原则

### 关键帧设计要点
1. **起止帧必须是中性姿态** (所有关节 = 0)
2. **脊椎运动幅度 ≤ 15-20°** — 避免与弹簧力对抗
3. **预备动作**：先做小的准备动作，再做主动作（更自然）
4. **停顿**：关键姿态保持 0.5-1.0s，让观者看清
5. **返回帧给足时间**：至少 0.6s 从极限姿态回到中性

### 待修复的动作 (下次继续)
- `thinking`: 右手需要到达下巴位置，目前 arm2/arm4 角度不够
- `clap`: 双手需要在身前合拢，目前仍偏向两侧
- `point_right`: 应该只有右手动，目前控制信号可能有问题

### 插值引擎
- **Cubic Hermite 样条** (Catmull-Rom tangent) — 速度连续，比线性插值自然
- 控制频率: 50Hz | 物理仿真: 500Hz

---

## 7. 离线渲染注意事项

```xml
<!-- 必须设置 offscreen framebuffer 尺寸 -->
<visual><global offwidth="1920" offheight="1080"/></visual>
```

```bash
# Linux 无显示器，必须用 EGL
MUJOCO_GL=egl python3 script.py
```

MuJoCo Python 安装位置: `/home/zero/mujoco_playground/.venv/`

---

## 8. 文件结构

```
models/guge/
├── scene_guge.xml            # MuJoCo 最终场景文件
├── MODEL_EXPERIENCE.md       # 本文档
└── videos/
    ├── joint_verification.mp4      # 关节逐个验证
    ├── joint_directions_+30deg.png # 关节方向参考图
    ├── azimuth_grid.png            # 8方位相机参考
    ├── motion_01_wave_hello.mp4    # 各动作独立视频
    ├── ...
    └── motion_all_demo.mp4         # 12动作合集

models/guge_urdf/
├── urdf/guge_urdf.urdf   # 原始 URDF
├── meshes/*.STL           # 网格文件
└── config/*.yaml          # 关节配置

controllers/
└── motion_library.py      # 动作库 + 样条插值引擎 (v2)

tools/
└── build_guge_scene.py    # 模型构建脚本

examples/
├── verify_guge_joints.py  # 关节验证脚本
└── record_guge_motions.py # 批量视频录制
```
