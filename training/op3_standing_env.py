"""
OpenServoSim - Custom MJX RL Environment for OP3 Standing Balance

A self-contained MjxEnv subclass that loads the local OP3 model
(models/reference/robotis_op3/scene_rl.xml) and trains a standing
balance policy using MJX + Brax PPO.

No dependency on mujoco_playground's registry — only uses its base
classes (mjx_env.MjxEnv, State) and wrapper utilities.
"""

import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

# Path to the OP3 model in OpenServoSim
_OPENSERVOSIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE_XML = os.path.join(
    _OPENSERVOSIM_ROOT, "models", "reference", "robotis_op3", "scene_rl.xml"
)

ROOT_BODY = "body_link"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=40.0,       # Enhanced servo stiffness
        Kd=1.084,       # Joint damping
        early_termination=True,
        action_repeat=1,
        action_scale=0.3,
        obs_noise=0.05,
        obs_history_size=3,
        reward_config=config_dict.create(
            scales=config_dict.create(
                upright=5.0,            # Stay vertical
                height=2.0,             # Maintain standing height
                lin_vel_z=-2.0,         # No bouncing
                ang_vel_xy=-0.5,        # No wobbling
                xy_drift=-1.0,          # Stay near origin
                torques=-0.0002,        # Energy efficiency
                action_rate=-0.01,      # Smooth motions
                energy=-0.0001,         # Power saving
                termination=-1.0,       # Don't fall
            ),
        ),
        impl="jax",
        naconmax=16 * 8192,
        njmax=16 * 4 + 20 * 4,
    )


class Op3StandingEnv(mjx_env.MjxEnv):
    """OP3 standing balance environment for RL training."""

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)

        # Load model from local XML with assets
        model_dir = os.path.join(
            _OPENSERVOSIM_ROOT, "models", "reference", "robotis_op3"
        )
        assets_dir = os.path.join(model_dir, "assets")

        # Build assets dict
        self._model_assets = {}
        # Load all XML files in model dir
        for f in os.listdir(model_dir):
            fpath = os.path.join(model_dir, f)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as fp:
                    self._model_assets[f] = fp.read()
        # Load mesh assets
        if os.path.isdir(assets_dir):
            for f in os.listdir(assets_dir):
                fpath = os.path.join(assets_dir, f)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as fp:
                        self._model_assets[f] = fp.read()
            # Simplified convex meshes
            convex_dir = os.path.join(assets_dir, "simplified_convex")
            if os.path.isdir(convex_dir):
                for f in os.listdir(convex_dir):
                    fpath = os.path.join(convex_dir, f)
                    if os.path.isfile(fpath):
                        with open(fpath, "rb") as fp:
                            self._model_assets[f] = fp.read()

        # Load XML
        xml_path = os.path.join(model_dir, "scene_rl.xml")
        with open(xml_path, "r") as fp:
            xml_string = fp.read()

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_string, assets=self._model_assets
        )
        self._mj_model.opt.timestep = config.sim_dt

        # Set PD gains
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase render resolution
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

        # Post-init
        self._init_q = jp.array(self._mj_model.keyframe("stand").qpos)
        self._default_pose = self._mj_model.keyframe("stand").qpos[7:]
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._torso_body_id = self._mj_model.body(ROOT_BODY).id
        self._init_torso_z = float(self._mj_model.keyframe("stand").qpos[2])

    # ---- Sensor helpers ----

    def _get_sensor(self, data: mjx.Data, name: str) -> jax.Array:
        return mjx_env.get_sensor_data(self._mj_model, data, name)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return self._get_sensor(data, "gyro")

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return self._get_sensor(data, "upvector")

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return self._get_sensor(data, "global_linvel")

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return self._get_sensor(data, "global_angvel")

    # ---- Environment interface ----

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, noise_rng = jax.random.split(rng)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=self._init_q,
            qvel=jp.zeros(self._mjx_model.nv),
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._mjx_model.nu),
            "last_last_act": jp.zeros(self._mjx_model.nu),
            "step": 0,
            "motor_targets": jp.zeros(self._mjx_model.nu),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs_dim = 46  # gyro(3) + gravity(3) + qpos_delta(20) + last_act(20)
        obs_history = jp.zeros(self._config.obs_history_size * obs_dim)
        obs = self._get_obs(data, info, obs_history, noise_rng)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, noise_rng = jax.random.split(state.info["rng"])

        motor_targets = self._default_pose + action * self._config.action_scale
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        data = mjx_env.step(
            self._mjx_model, state.data, motor_targets, self.n_substeps
        )

        obs = self._get_obs(data, state.info, state.obs, noise_rng)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Bookkeeping
        state.info["motor_targets"] = motor_targets
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step"] += 1
        state.info["rng"] = rng

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = jp.float32(done)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        torso_z = data.xpos[self._torso_body_id, -1]
        gravity_z = self.get_gravity(data)[-1]

        fall = gravity_z < 0.7
        fall |= torso_z < 0.18

        joint_angles = data.qpos[7:]
        joint_limit_exceed = jp.any(joint_angles < self._lowers)
        joint_limit_exceed |= jp.any(joint_angles > self._uppers)

        return jp.where(
            self._config.early_termination,
            fall | joint_limit_exceed,
            joint_limit_exceed,
        )

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        obs_history: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        obs = jp.concatenate([
            self.get_gyro(data),                       # 3
            self.get_gravity(data),                    # 3
            data.qpos[7:] - self._default_pose,        # 20
            info["last_act"],                           # 20
        ])  # total = 46

        if self._config.obs_noise >= 0.0:
            noise = self._config.obs_noise * jax.random.uniform(
                rng, obs.shape, minval=-1.0, maxval=1.0
            )
            obs = jp.clip(obs, -100.0, 100.0) + noise

        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        gravity = self.get_gravity(data)
        global_linvel = self.get_global_linvel(data)
        global_angvel = self.get_global_angvel(data)
        torso_z = data.xpos[self._torso_body_id, -1]

        return {
            # Stay upright (gravity z should be close to 1.0)
            "upright": jp.exp(-jp.sum(jp.square(gravity[:2])) * 10.0),
            # Maintain standing height
            "height": jp.exp(-jp.square(torso_z - self._init_torso_z) * 100.0),
            # No bouncing
            "lin_vel_z": jp.square(global_linvel[2]),
            # No wobbling
            "ang_vel_xy": jp.sum(jp.square(global_angvel[:2])),
            # Stay near origin
            "xy_drift": jp.sum(jp.square(data.qpos[:2])),
            # Energy efficiency
            "torques": jp.sqrt(jp.sum(jp.square(data.actuator_force)))
            + jp.sum(jp.abs(data.actuator_force)),
            # Smooth actions
            "action_rate": jp.sum(
                jp.square(action - info["last_act"])
            ) + jp.sum(
                jp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            # Power saving
            "energy": jp.sum(
                jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)
            ),
            # Don't fall
            "termination": done & (info["step"] < 500),
        }

    # ---- Properties ----

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
