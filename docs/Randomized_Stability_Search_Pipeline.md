# 🧬 Data-Driven Gait Robustness Tuning Pipeline

*A systematic methodology for discovering highly stable robotic gaits through randomized headless simulation ("The 100-Iteration Crucible").*

---

## 📌 1. The Core Problem: Why Manual Tuning Fails
When developing walking gaits (like the ROBOTIS sinusoidal gait combined with UVC balance), parameters are usually manually tuned while the robot walks in a straight line. 
However, **omnidirectional teleoperation is chaotic**. When a human user abruptly switches between Forward, Sidestep, and Turning, the robot experiences sudden inertial shifts. These combinations create compound errors in the robot's center of mass (COM) that manual tuning simply cannot predict or cover.

**The Solution:** Instead of guessing parameters, we define a search space, simulate a brutal "stress test" sequence, and let the physics engine find the dominant survivor.

---

## ⚙️ 2. The Methodology

The pipeline (as implemented in `examples/07_randomized_stability_test.py`) consists of four core pillars:

### Pillar A: The Search Space (Mutation)
We take a baseline gait (e.g., the `ninja` preset) and apply random mutations to all physical parameters. 

| Parameter Group | Examples | Why Mutate? |
| :--- | :--- | :--- |
| **Gait Timing** | `period_time`, `dsp_ratio` | Faster cadence vs. longer ground support. |
| **Kinematics** | `init_z_offset`, `x_move_amplitude` | Lower center of gravity vs. wider stance. |
| **Balance Reflex** | `balance_ankle_pitch_gain` | How hard the ankles fight to keep the foot flat. |
| **UVC Posture** | `gain_roll`, `gain_pitch` | How aggressively the torso tilts to fight momentum. |

*Crucial rule: Always set reasonable bounds (`np.random.uniform(min, max)`) so the engine doesn't test physically impossible configurations.*

### Pillar B: The Stress Test (The Crucible)
The robot must run a **deterministic sequence of chaotic inputs**. It cannot be random, otherwise, we cannot compare Score A to Score B fairly.
A good stress test sequence mimics a panicked human controller:
1.  **`t=0-2s`**: Idle / Stabilize (Let the robot drop into its crouch).
2.  **`t=2-4s`**: Sprint Forward + Turn Left simultaneously (Centrifugal force test).
3.  **`t=4-6s`**: Sudden Sprint Backward + Sidestep Right (Momentum reversal test).
4.  **`t=6-8s`**: Maximum speed forward push.
5.  **`t=8-10s`**: Heavy spin in place.

### Pillar C: The Fitness Function (Scoring)
How do we know which mutated parameter set "won"? We need a math formula to rank them.

```python
if body_height < 0.17m:
    # THE ROBOT FELL
    score = survival_time  # E.g., fell at 4.2 seconds -> Score = 4.2
else:
    # THE ROBOT SURVIVED THE FULL 10 SECONDS
    # 1. Dist Reward: Did it actually move, or just safely step in place?
    dist_reward = path_length * 20.0 
    
    # 2. Height Reward: Did it keep its hips high, or dragging on the floor?
    height_reward = min_height * 10.0
    
    # 3. Variance Penalty: Was it shaking wildly to survive?
    instability = (pitch_variance * 0.1) + (roll_variance * 0.1)
    
    # Base 10.0 points just for surviving
    score = 10.0 + (dist_reward + height_reward) / (1.0 + instability)
```

### Pillar D: Massive Headless Execution
We run `N` iterations (usually 100 to 500) through MuJoCo's ultra-fast headless calculation. 
- *Why headless?* Removing the GUI renderer (`mujoco.viewer`) allows the CPU to process 10 seconds of physics in ~0.8 seconds.
- 100 iterations of 10-second chaotic testing only takes ~2 minutes of wall-clock time.

---

## 🚀 3. How to Use This for Future Gait Development

Whenever you write a **new walking engine** (e.g., transitioning to a reinforcement learning model, MPC, or a new IK solver), immediately build an `examples/xx_randomized_test.py` script for it.

**Workflow:**
1. Get the new algorithm barely walking (the "Baseline").
2. Write a Stress Test sequence that explicitly tries to knock the Baseline over.
3. Define the mutable variables as a search space.
4. Run 100 iterations overnight or during a coffee break.
5. Extract `Iteration #X` (the highest scorer).
6. Hardcode `Iteration #X`'s exact parameters as your new default preset.

This method transforms qualitative tuning ("it feels a bit unstable") into quantitative optimization ("Iteration 43 pushed pitch variance down by 40% while surviving the turn").
