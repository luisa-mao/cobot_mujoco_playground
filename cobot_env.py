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
ROBOT_XML = "/home/luisamao/villa_spaces/sim_ws/robot_simple_collision_ee.xml"
TABLE_XML = "/home/luisamao/villa_spaces/sim_ws/table.xml"


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      nworld=1024,
      cam_res=(256, 256),
      use_textures=True,
      use_shadows=False,
      render_rgb=(True, True),
      render_depth=(False, False),
      enabled_geom_groups=[0, 1, 2, 3],
      cam_active=(False, False, True, True), # [sidecam, topdown, basecam, handcam] ?
  )


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02, 
      sim_dt=0.0005,
      episode_length=250,
      action_repeat=1,
      vision=False,
      vision_config=default_vision_config(),
      impl="warp",
      naconmax=20_000,
      njmax=30_000,
      naccdmax=5000,
      num_blocks = 3,
      num_joints = 9,
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
        z_pos = table_surface_z + (block_size[2] * 2 * i) + block_size[2] + 0.01 + (i * 0.005)
        
        block_name = f'block_{i}'
        block_pos = [base_x, base_y, z_pos]
        
        # Create unique body
        block_body = ET.SubElement(worldbody, 'body', name=block_name, 
                                   pos=' '.join(map(str, block_pos)), childclass='interactive')

        # Each block needs its own freejoint to move independently
        # ET.SubElement(block_body, 'freejoint', name=f'joint_{block_name}')
        ET.SubElement(block_body, 'joint', type='free', name=f'joint_{block_name}')
        
        # Add geom with stable manipulation parameters
        ET.SubElement(block_body, 'geom', name=f'geom_{block_name}', type='box', 
                      size=' '.join(map(str, block_size)), 
                      rgba='1 0 0 1' if i % 2 == 0 else '0 1 0 1', # Alternate colors
                      mass='0.1', friction='1.0 0.01 0.0001', condim='4',
                      solimp="0.9 0.95 0.001", solref="0.01 1.0")

    # 4. Inject Stability Class & Missing Materials
    if root.find(".//default[@class='interactive']") is None:
        interactive = ET.SubElement(default_elem, 'default', {'class': 'interactive'})
        ET.SubElement(interactive, 'joint', armature='0.05', damping='1.0', frictionloss='0.01')

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

        # model stuff
        self._num_blocks: int = self._config.num_blocks
        self._xml_string = prepare_cobot_model(ROBOT_XML, TABLE_XML, ASSETS_DIR, num_blocks = self._num_blocks)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml_string)
        self._num_joints: int = self._config.num_joints
        self._action_scale = 0.1
        self._episode_length: int = self._config.episode_length
        # print(f"\nModel loaded successfully!")
        # mj_model = self._mj_model
        # print(f"  - DOFs: {mj_model.nv}")
        # print(f"  - Bodies: {mj_model.nbody}")
        # print(f"  - Joints: {mj_model.njnt}")
        # print(f"  - Actuators: {mj_model.nu}")
        # print("geom groups", self._mj_model.geom_group)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mj_model.opt.iterations = 60
        self._mj_model.opt.ls_iterations = 25
        self._mj_model.opt.ccd_iterations = 600
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._table_idx = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
        self._ee_site_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pinch_site'
        )
        self._gripper_max_width = 0.04
        self._difficulty_buckets = 10


        for i in range(self._mj_model.njnt):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            start_idx = self._mj_model.jnt_qposadr[i]
            # Check joint type to know the width (Free=7, Ball=4, Hinge/Slide=1)
            jnt_type = self._mj_model.jnt_type[i]
            width = {0: 7, 1: 4, 2: 1, 3: 1}[jnt_type] 
            print(f"Joint {i} [{name}]: qpos indices {start_idx} to {start_idx + width}")

        # vision stuff
        if self._vision:
            vision_kwargs = self._config.vision_config.to_dict()
            self._rc = mjx.create_render_context(
                mjm=self._mj_model,
                **vision_kwargs
            )
            self._rc_pytree = self._rc.pytree()
            self._wristcam_idx = 1
            self._basecam_idx = 0

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = jp.zeros(self.mjx_model.nq) 
        qvel = jp.zeros(self.mjx_model.nv) 

        # jax.debug.print("Resetting environment with {num_blocks} blocks, {num_joints} joints, {qpos} qpos.", num_blocks=self._num_blocks, num_joints=self._num_joints, qpos=qpos.shape)

        # table and blocks
        table_surface_z = self.mjx_model.geom_pos[self._table_idx, 2]
        block_half_height = 0.03
        rng, pos_key = jax.random.split(rng)
    
        block_base_x, block_base_y = 0.6, -0.18
        jitter = 0.1
        # block_base_x, block_base_y = 0.72, -0.00
        # jitter = 0.01
        xy_offset = jax.random.uniform(
            pos_key, (2,), minval=-jitter, maxval=jitter
        )
        block_curr_x = block_base_x + xy_offset[0] * 0.1
        block_curr_y = block_base_y + xy_offset[1]
        for i in range(self._num_blocks):
            start_idx = self._num_joints + (i * 7)
            z_pos = table_surface_z + (block_half_height * 2 * i) + block_half_height + 0.04 + (i * 0.01)
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
        new_geom_xpos = data.geom_xpos.at[self._table_idx, 2].set(table_surface_z)
        data = data.replace(geom_xpos=new_geom_xpos)
        data = mjx.forward(self.mjx_model, data)

        # settle
        def settle_fn(i, val):
            return mjx.step(self.mjx_model, val)
        
        data = jax.lax.fori_loop(0, 200, settle_fn, data)

        # set robot after settling
        # robot
        # robot_home = jp.array([0.2, 1.019, 0.144, .6, -0.221, 0.7, -0.886, 0.04])
        # robot_home = jp.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        robot_home = jp.array([
                -2.76968753372131e-05,
                0.2623287909343436,
                -3.1403614742178494,
                -2.269397010260591,
                -0.0003360909295677672,
                0.9596066489970824,
                1.5707649014940337,
                0, 0
            ]            
        )
        new_qpos = data.qpos.at[:self._num_joints].set(robot_home)
        new_qvel = data.qvel.at[:self._num_joints].set(jp.zeros(self._num_joints))
        data = data.replace(qpos=new_qpos, qvel=new_qvel)
        data = mjx.forward(self.mjx_model, data)
        # todo: randomize block poses

        metrics = {
            'reward/xy_displace': jp.zeros(()),
            'reward/top_displace': jp.zeros(()),
            'reward/table_penalty': jp.zeros(()),
            'reward/reward_approach': jp.zeros(()),
            'reward/reward_top_knockdown': jp.zeros(()),
            'reward/reward_bottom_knockdown': jp.zeros(()),
            'reward/reward': jp.zeros(()),
            'success': jp.zeros(()),
            'success_rate': jp.zeros(()),
            'reward/reward_action_rate': jp.zeros(()),
            'reward/reward_action_direction': jp.zeros(()),
            'blocks_fell': jp.zeros(())
        }

        info = {
            'prev_ctrl': jp.zeros(self._num_joints - 1),
            'prev_action': jp.zeros(self._num_joints - 1),
            'step': jp.zeros(()),
        }
        for i in range(self._num_blocks):
            start_idx = self._num_joints + (i * 7)
            info[f'block{i+1}_init_pos'] = data.qpos[start_idx : start_idx + 3]
        
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)

        # vision
        if self._vision:
            obs = {"joint_states": obs}
            render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
            out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
            basecam_rgb = mjx.get_rgb(self._rc_pytree, self._basecam_idx, out[0])
            wristcam_rgb = mjx.get_rgb(self._rc_pytree, self._wristcam_idx, out[0])
            info["basecam_frames"] = basecam_rgb
            info["wristcam_frames"] = wristcam_rgb
            obs["pixels/basecam"] = basecam_rgb
            obs["pixels/wristcam"] = wristcam_rgb
    
        state = mjx_env.State(data, obs, reward, done, metrics, info)
        return state
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl = self._get_ctrl(state.data, action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        r, done, metrics = self._get_reward(data, action, state.info, state.metrics)
        obs = self._get_obs(data, state.info)

        info = dict(state.info)
        info['prev_ctrl'] = ctrl
        info['prev_action'] = action
        info['step'] = info.get('step', jp.zeros(())) + 1

        if self._vision:
            obs = {"joint_states": obs}
            render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
            out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
            basecam_rgb = mjx.get_rgb(self._rc_pytree, self._basecam_idx, out[0])
            wristcam_rgb = mjx.get_rgb(self._rc_pytree, self._wristcam_idx, out[0])
            info["basecam_frames"] = basecam_rgb
            info["wristcam_frames"] = wristcam_rgb
            obs["pixels/basecam"] = basecam_rgb
            obs["pixels/wristcam"] = wristcam_rgb

        # return mjx_env.State(data, obs, r, done, metrics, info)
        return state.replace(
            data=data, 
            obs=obs, 
            reward=r,      # Note: The field name in State is 'reward', not 'r'
            done=done, 
            metrics=metrics,
            info=info
        )


    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        qpos_raw = data.qpos[:self._num_joints]
        qpos_arm_normalized = (qpos_raw + jp.pi) % (2 * jp.pi) - jp.pi
        prev_action = info['prev_action']
        
        # Velocities are usually fine as raw values (rad/s)
        qvel_arm = data.qvel[:self._num_joints]
        ee_pos = data.site_xpos[self._ee_site_idx]
        
        if self._vision:
            obs = jp.concatenate([
                qpos_arm_normalized, # self._num_joints dimensions
                prev_action,
                qvel_arm,        # self._num_joints dimensions
                # ee_pos,          # 3 dimensions
            ])
        else:
            block_positions = [
                data.qpos[self._num_joints + (i * 7) : self._num_joints + (i * 7) + 3] 
                for i in range(self._num_blocks)
            ]

            # 2. Concatenate the arm data with the dynamic list of block positions
            obs = jp.concatenate([
                qpos_arm_normalized,  # self._num_joints dims
                prev_action,      # self._num_joints dims
                qvel_arm,         # self._num_joints dims
                # ee_pos,           # 3 dims
                *block_positions  # Unpacks the list of (3-dim) arrays into the concatenation
            ])
        return jp.clip(obs, -10.0, 10.0)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> Tuple[jax.Array, jax.Array, dict[str, Any]]:
        
        hand_pos = data.site_xpos[self._ee_site_idx]
        safety_margin = 0.03
        table_z = self.mjx_model.geom_pos[self._table_idx, 2]
        penetration = table_z + safety_margin - hand_pos[2]
        penalty_table = jp.where(penetration > 0, -5.0, 0.0)
        penalty_action_rate = jp.linalg.norm(action * self._action_scale)
        penalty_action_direction = jp.linalg.norm((action - info.get('prev_action', jp.zeros_like(action))) * self._action_scale) * -1.0
        
        if self._num_blocks == 0: # penalty action rate should be negative here...
            reward = jp.zeros(())
            done = jp.zeros(())
            metrics = dict(metrics)
            metrics["reward/xy_displace"] = 0.0
            metrics["reward/top_displace"] = 0.0
            metrics["reward/table_penalty"] = penalty_table
            metrics["reward/reward_approach"] = 0.0
            metrics["reward/reward_top_knockdown"] = 0.0
            metrics["reward/reward_bottom_knockdown"] = 0.0
            metrics["reward/reward_action_rate"] = penalty_action_rate
            metrics["reward/reward_action_direction"] = penalty_action_direction
            metrics["reward/reward"] = reward
            metrics["success"] = 0.0
            metrics["success_rate"] = 0.0
            metrics["blocks_fell"] = 0.0
            return reward, done, metrics

        current_blocks_pos = jp.stack([
            data.qpos[self._num_joints + (i * 7) : self._num_joints + (i * 7) + 3] 
            for i in range(self._num_blocks)
        ])
        initial_blocks_pos = jp.stack([
            info[f'block{i+1}_init_pos'] 
            for i in range(self._num_blocks)
        ])

        # Shape: (num_blocks,)
        xy_move = jp.linalg.norm(current_blocks_pos[:, :2] - initial_blocks_pos[:, :2], axis=-1)
        z_drop = initial_blocks_pos[:, 2] - current_blocks_pos[:, 2]
        
        other_xy_move = xy_move[:-1]
        other_z_drop = z_drop[:-1]

        is_failure = jp.any(other_xy_move > 0.03) | jp.any(other_z_drop > 0.03)

        top_xy_move = xy_move[-1]
        top_z_drop = z_drop[-1]
        top_pos = current_blocks_pos[-1]

        is_success = (top_z_drop > 0.06) & (top_xy_move > 0.03)
        top_reward_knockdown = jp.where(is_success, 20.0, 0.0)
        reward_bottom_knockdown = jp.where(is_failure, -10.0, 0.0)
        # top_reward_knockdown = jp.where(is_success, 80.0, 0.0)
        # reward_bottom_knockdown = jp.where(is_failure, -100.0, 0.0)
        reward_knockdown = top_reward_knockdown + reward_bottom_knockdown

        blocks_fell = (is_success | is_failure).astype(jp.float32)
        done = jp.zeros(())

        penalty_action_rate = penalty_action_rate * (-1.0 + blocks_fell * -10.0)
        dist_to_top_block = jp.linalg.norm(hand_pos - top_pos)
        # amplitude = 4.0
        # sigma = 0.1
        # reward_approach_pos = amplitude * jp.exp(-(dist_to_top_block**2) / (2 * sigma**2))
        # reward_approach = (-dist_to_top_block + reward_approach_pos) * (1.0 - blocks_fell)
        reward_approach = (-dist_to_top_block * 2.0) * (1.0 - blocks_fell)
        
        reward = reward_knockdown + reward_approach + penalty_table + penalty_action_rate + penalty_action_direction + abs(top_xy_move) * 10.0
        
        metrics = dict(metrics)
        metrics["reward/xy_displace"] = abs(top_xy_move)
        metrics["reward/top_displace"] = top_z_drop
        metrics["reward/table_penalty"] = penalty_table
        metrics["reward/reward_approach"] = reward_approach
        metrics["reward/reward_top_knockdown"] = top_reward_knockdown
        metrics["reward/reward_bottom_knockdown"] = reward_bottom_knockdown
        metrics["reward/reward_action_rate"] = penalty_action_rate
        metrics["reward/reward_action_direction"] = penalty_action_direction
        metrics["reward/reward"] = reward
        metrics["success"] = (is_success & ~is_failure).astype(jp.float32)
        metrics["success_rate"] = (is_success & ~is_failure & (info.get('step', jp.zeros(())) >= self._episode_length - 1)).astype(jp.float32)
        metrics["blocks_fell"] = blocks_fell

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
        ctrl_delta = jp.concat([action[:-1] * self._action_scale, action[-1:] * self._gripper_max_width])
        joint_state = data.qpos[:self._num_joints - 1]
        ctrl = joint_state + ctrl_delta

        arm_ctrl = ctrl[:7]
        gripper_ctrl = ctrl[7:8] # Using slice [7:8] keeps it as an array of size 1
        arm_wrapped = (arm_ctrl + jp.pi) % (2 * jp.pi) - jp.pi
        full_ctrl = jp.concatenate([arm_wrapped, gripper_ctrl])

        # Apply Hard Physical Limits to the Bounded Joints (1, 3, 5)
        # We define the limits for those specific indices
        low_lim = jp.array([-2.24, -2.57, -2.09])
        high_lim = jp.array([2.24, 2.57, 2.09])
        lim_idx = jp.array([1, 3, 5])
        
        # Clip only the limited joints to ensure they stay within MuJoCo's bounds
        final_ctrl = full_ctrl.at[lim_idx].set(
            jp.clip(full_ctrl[lim_idx], low_lim, high_lim)
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
