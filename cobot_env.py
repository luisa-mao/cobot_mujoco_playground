from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
import xml.etree.ElementTree as ET
import os

ASSETS_DIR = "/home/luisamao/villa_spaces/sim_ws/src/mujoco_cobot/assets"
ROBOT_XML = "/home/luisamao/villa_spaces/sim_ws/robot_simple_collision.xml"
TABLE_XML = "/home/luisamao/villa_spaces/sim_ws/table.xml"


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      nworld=1024,
      cam_res=(64, 64),
      use_textures=False,
      use_shadows=False,
      render_rgb=(True,),
      render_depth=(False,),
      enabled_geom_groups=[0, 1, 2],
      cam_active=(True, False), # [fixed, lookatcart]
  )


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      vision_config=default_vision_config(),
      impl="warp",
      naconmax=50_000,
      njmax=500,
      naccdmax=2500, # where to put this one
  )

def prepare_cobot_model(robot_xml_path, table_xml_path, assets_dir, num_blocks=3):
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    # 1. Setup Asset/Default elements
    asset_elem = root.find('asset') or ET.SubElement(root, 'asset')
    default_elem = root.find('default') or root.insert(0, ET.Element('default')) or root.find('default')

    # 2. Inject Table
    ET.SubElement(worldbody, 'include', file=table_xml_path)
    
    # 3. Inject Stack of Blocks
    block_size = [0.03, 0.03, 0.03]  # Half-extents
    table_surface_z = 0.72
    base_x, base_y = 0.8, 0.0
    
    for i in range(num_blocks):
        # Calculate Z: surface + (2 * half_height * index) + offset
        # We add 0.01 initial gap from table, then 0.001 between blocks
        z_pos = table_surface_z + (block_size[2] * 2 * i) + block_size[2] + 0.01 + (i * 0.001)
        
        block_name = f'block_{i}'
        block_pos = [base_x, base_y, z_pos]
        
        # Create unique body
        block_body = ET.SubElement(worldbody, 'body', name=block_name, 
                                   pos=' '.join(map(str, block_pos)), childclass='interactive')
        
        # Each block needs its own freejoint to move independently
        ET.SubElement(block_body, 'freejoint', name=f'joint_{block_name}')
        
        # Add geom with stable manipulation parameters
        ET.SubElement(block_body, 'geom', name=f'geom_{block_name}', type='box', 
                      size=' '.join(map(str, block_size)), 
                      rgba='1 0 0 1' if i % 2 == 0 else '0 1 0 1', # Alternate colors
                      mass='0.1', friction='1.5 0.01 0.0001', condim='4',
                      solref='0.02 1.0', solimp='0.9 0.95 0.001')

    # 4. Inject Stability Class & Missing Materials
    if root.find(".//default[@class='interactive']") is None:
        interactive = ET.SubElement(default_elem, 'default', {'class': 'interactive'})
        ET.SubElement(interactive, 'joint', armature='0.01', damping='0.05')

    if root.find(".//material[@name='gym_floor_mat']") is None:
        ET.SubElement(asset_elem, 'texture', name='gym_floor_tex', type='2d', builtin='checker', 
                      rgb1='.2 .3 .4', rgb2='.1 .2 .3', width='512', height='512')
        ET.SubElement(asset_elem, 'material', name='gym_floor_mat', texture='gym_floor_tex', 
                      texrepeat='2 2', specular='0.3', shininess='0.5')

    # 5. Path Resolution
    for elem in root.iter():
        if 'file' in elem.attrib:
            path = elem.attrib['file']
            if not os.path.isabs(path):
                elem.attrib['file'] = os.path.normpath(os.path.join(assets_dir, path))

    return ET.tostring(root, encoding='unicode')


class CobotEnv(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        self._vision = self._config.vision
        self._config.episode_length = 250

        # model stuff
        self._xml_string = prepare_cobot_model(ROBOT_XML, TABLE_XML, ASSETS_DIR)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml_string)
        print(f"\nModel loaded successfully!")
        mj_model = self._mj_model
        print(f"  - DOFs: {mj_model.nv}")
        print(f"  - Bodies: {mj_model.nbody}")
        print(f"  - Joints: {mj_model.njnt}")
        print(f"  - Actuators: {mj_model.nu}")
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 30
        mj_model.opt.ls_iterations = 10
        self._mj_model.opt.ccd_iterations = 70
        # self._mj_model.opt.naccdmax = 512 
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._table_idx = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
        self._ee_site_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pinch_site'
        )


        # vision stuff
        # vision_kwargs = self._config.vision_config.to_dict()
        # self._rc = mjx.create_render_context(
        #     mjm=self._mj_model,
        #     **vision_kwargs
        # )
        # self._rc_pytree = self._rc.pytree()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = jp.zeros(self.mjx_model.nq) 
        qvel = jp.zeros(self.mjx_model.nv) 

        # robot
        robot_home = jp.array([0.2, 1.019, 0.144, .6, -0.221, 0.5, -0.886])
        qpos = qpos.at[:7].set(robot_home)

        # table and blocks
        table_surface_z = self.mjx_model.geom_pos[self._table_idx, 2] + 0.02
        block_half_height = 0.03
        num_blocks = (self.mjx_model.nq - 7) // 7  # Calculate N based on qpos size

        rng, pos_key = jax.random.split(rng)
    
        block_base_x, block_base_y = 0.8, 0.0
        jitter = 0.0 # 0.1
        xy_offset = jax.random.uniform(
            pos_key, (2,), minval=-jitter, maxval=jitter
        )
        block_curr_x = block_base_x + xy_offset[0]
        block_curr_y = block_base_y + xy_offset[1]
        for i in range(num_blocks):
            start_idx = 7 + (i * 7)
            z_pos = table_surface_z + (block_half_height * 2 * i) + block_half_height + 0.02 + (i * 0.001)
            block_state = jp.array([block_curr_x, block_curr_y, z_pos, 1.0, 0.0, 0.0, 0.0])
            qpos = qpos.at[start_idx : start_idx + 7].set(block_state)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
            naccdmax=self._config.naccdmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # settle
        def settle_fn(i, val):
            return mjx.step(self.mjx_model, val)
        
        data = jax.lax.fori_loop(0, 200, settle_fn, data)
        data = mjx.forward(self.mjx_model, data)

        # todo: randomize block poses

        # metrics
        metrics = {
            'reward/top_displace': jp.zeros(()),
            'reward/table_penalty': jp.zeros(()),
            'reward/dist_to_block': jp.zeros(()),
            'reward/reward_top_knockdown': jp.zeros(()),
            'reward/reward_bottom_knockdown': jp.zeros(()),
            'reward/reward': jp.zeros(()),
            'success': jp.zeros(()),
            'reward/reward_action_rate': jp.zeros(()),
        }

        info = {
            'prev_ctrl': jp.zeros(7),
            'prev_action': jp.zeros(7),
            'block1_init_pos': data.qpos[7:10],  # Position of the first block in the stack
            'block2_init_pos': data.qpos[14:17],  # Position of the second block in the stack
            'block3_init_pos': data.qpos[21:24],  # Position of the third block in the stack
        }
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)

        # vision
        # render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        # out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
        # rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
        # gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
        # frame_stack = jp.repeat(gray, 3, axis=-1)
        # info["frame_stack"] = frame_stack
        # obs = {"pixels/view_0": frame_stack}
 
        state = mjx_env.State(data, obs, reward, done, metrics, info)
        return state
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl = self._get_ctrl(state.data, action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        r, done, metrics = self._get_reward(data, action, state.info, state.metrics) # todo: implement

        obs = self._get_obs(data, state.info)
        # render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        # out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
        # rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
        # gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
        # prev_stack = state.info["frame_stack"]
        # frame_stack = jp.concatenate([prev_stack[..., 1:], gray], axis=-1)
        # info = dict(state.info)
        # info["frame_stack"] = frame_stack
        # info["time_out"] = done
        # obs = {"pixels/view_0": frame_stack}
        info = dict(state.info)
        info['prev_ctrl'] = ctrl
        info['prev_action'] = action
        return mjx_env.State(data, obs, r, done, metrics, info)


    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        qpos_arm_smooth = data.qpos[:7]
        prev_action = info['prev_action']
        
        # Velocities are usually fine as raw values (rad/s)
        qvel_arm = data.qvel[:7]
        
        # EE and Block positions
        ee_pos = data.site_xpos[self._ee_site_idx]
        block_0_pos = data.qpos[7:10]
        block_1_pos = data.qpos[14:17]
        block_2_pos = data.qpos[21:24]

        obs = jp.concatenate([
            qpos_arm_smooth, # 7 dimensions
            prev_action,
            qvel_arm,        # 7 dimensions
            ee_pos,          # 3 dimensions
            block_0_pos,     # 3 dimensions
            block_1_pos,     # 3 dimensions
            block_2_pos      # 3 dimensions
        ])
        return jp.clip(obs, -10.0, 10.0)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> Tuple[jax.Array, jax.Array, dict[str, Any]]:
        pos_0 = data.qpos[7:10]
        pos_1 = data.qpos[14:17]
        pos_2 = data.qpos[21:24]
        hand_pos = data.site_xpos[self._ee_site_idx]

        pos_0_init = info['block1_init_pos']
        pos_1_init = info['block2_init_pos']
        pos_2_init = info['block3_init_pos']

        move_0 = jp.linalg.norm(pos_0[:2] - pos_0_init[:2])
        move_1 = jp.linalg.norm(pos_1[:2] - pos_1_init[:2])
        is_failure = (move_0 > 0.03) | (move_1 > 0.03)

        z_dropped = pos_2_init[2] - pos_2[2]
        is_success = z_dropped > 0.04 # Fell at least 8cm down

        safety_margin = 0.05 
        table_z = self.mjx_model.geom_pos[self._table_idx, 2]
        penetration = table_z + safety_margin - hand_pos[2]
        penalty_table = jp.where(penetration > 0, -10.0, 0.0)

        top_reward_knockdown = jp.where(is_success, 80.0, 0.0)
        reward_bottom_knockdown = jp.where(is_failure, -100.0, 0.0)
        reward_knockdown = top_reward_knockdown + reward_bottom_knockdown

        done = (is_success | is_failure).astype(jp.float32)
        penalty_action_rate = jp.linalg.norm(action * 0.01) * -0.5
        dist_to_top_block = jp.linalg.norm(hand_pos - pos_2)
        reward_approach = -dist_to_top_block
        
        reward = reward_knockdown + reward_approach + penalty_table + penalty_action_rate
        
        metrics = dict(metrics)
        metrics["reward/top_displace"] = z_dropped
        metrics["reward/table_penalty"] = penalty_table
        metrics["reward/dist_to_block"] = dist_to_top_block
        metrics["reward/reward_top_knockdown"] = top_reward_knockdown
        metrics["reward/reward_bottom_knockdown"] = reward_bottom_knockdown
        metrics["reward/reward_action_rate"] = penalty_action_rate
        metrics["reward/reward"] = reward
        metrics["success"] = (is_success & ~is_failure).astype(jp.float32)

        # # Add this right before the return statement
        # jax.debug.print(
        #     "--- REWARD DEBUG --- "
        #     "\nReward: {rw} | Done: {d}"
        #     "\nz_drop: {zd} | Table_Pen: {tp} | Dist_Block: {db}"
        #     "\nKnock_Top: {kt} | Knock_Bot: {kb} | Act_Rate: {ar}",
        #     rw=reward,
        #     d=done,
        #     zd=z_dropped,
        #     tp=penalty_table,
        #     db=dist_to_top_block,
        #     kt=top_reward_knockdown,
        #     kb=reward_bottom_knockdown,
        #     ar=penalty_action_rate
        # )

        return reward, done, metrics

    def _get_ctrl(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        ctrl_delta = action * 0.1
        joint_state = data.qpos[:7]
        ctrl = joint_state + ctrl_delta

        wrapped_ctrl = (ctrl + jp.pi) % (2 * jp.pi) - jp.pi
    
        # Apply Hard Physical Limits to the Bounded Joints (1, 3, 5)
        # We define the limits for those specific indices
        low_lim = jp.array([-2.24, -2.57, -2.09])
        high_lim = jp.array([2.24, 2.57, 2.09])
        lim_idx = jp.array([1, 3, 5])
        
        # Clip only the limited joints to ensure they stay within MuJoCo's bounds
        final_ctrl = wrapped_ctrl.at[lim_idx].set(
            jp.clip(wrapped_ctrl[lim_idx], low_lim, high_lim)
        )
        return final_ctrl
    @property
    def xml_path(self) -> str:
        # Pylance needs this. You can return the base robot path.
        return ROBOT_XML    
        
    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
