import os
import functools
import time
from datetime import datetime
from typing import Any, Dict, Sequence, Tuple, Union, Callable, NamedTuple, Optional, List

# --- Environment Setup ---
# Set GPU and XLA flags for performance
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# --- Imports ---
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx

# Brax & Training
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

# Visualization & Utils
import mediapy as media
import matplotlib.pyplot as plt
from ml_collections import config_dict
from flax import struct

import xml.etree.ElementTree as ET
import wandb
import numpy as np


# Standardize numpy printing
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# --- Sanity Check ---
try:
    mujoco.MjModel.from_xml_string('<mujoco/>')
    print(f'MuJoCo/MJX Initialization successful. Using Device: {jax.devices()}')
except Exception as e:
    print(f'Initialization failed: {e}')


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


import functools
import warnings
import jax
import jax.numpy as jp
from datetime import datetime
from brax import envs
from brax.envs.base import Env, PipelineEnv, State  # The fix for Brax 0.14.1
from brax.training.agents.ppo import train as ppo
import mujoco

from brax.io import mjcf

class CobotEnv(PipelineEnv):
    def __init__(self, target_pos=None, **kwargs):
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 30     # Increase from 6
        mj_model.opt.ls_iterations = 10  # Important for stability
        mj_model.opt.timestep = 0.002

        sys = mjcf.load_model(mj_model)

        n_frames = 10
        table_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
        # jax.debug.print(f"Table geom index: {table_id}, Z position: {mj_model.geom_pos[table_id, 2]:.3f}")
        self._table_id = table_id
        super().__init__(sys, n_frames=n_frames, backend='mjx', **kwargs)

        self._ee_site_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pinch_site'
        )

        self._fixed_target = None
        if target_pos is not None:
            self._fixed_target = jp.array(target_pos)

    @property
    def action_size(self):
        return 7

    def reset(self, rng: jp.ndarray) -> State:
        # 1. Initialize full arrays
        qpos = jp.zeros(self.sys.nq) 
        qvel = jp.zeros(self.sys.nv) 
        
        # 2. Set Robot Home (Indices 0-6)
        robot_home = jp.array([0.2, 1.019, 0.144, .6, -0.221, 0.5, -0.886])
        qpos = qpos.at[:7].set(robot_home)
        
        # 3. Set Block Stack (Indices 7 to 7 + 7*N)
        table_surface_z = self.sys.geom_pos[self._table_id, 2] + 0.02
        block_half_height = 0.03
        num_blocks = (self.sys.nq - 7) // 7  # Calculate N based on qpos size

        rng, pos_key = jax.random.split(rng)
    
        block_base_x, block_base_y = 0.8, 0.0
        jitter = 0.1
        xy_offset = jax.random.uniform(
            pos_key, (2,), minval=-jitter, maxval=jitter
        )
        block_curr_x = block_base_x + xy_offset[0]
        block_curr_y = block_base_y + xy_offset[1]
        # print the table height from the sys model
        # jax.debug.print("Reset: Table Z from sys model shape: {s}", s=self.sys.geom_pos.shape)
        # jax.debug.print("Reset: Table Z from sys model: {z}", z=self.sys.geom_pos[self._table_id, 2])        
        # We loop to create a stack in qpos
        for i in range(num_blocks):
            # Calculate start index for this specific block
            start_idx = 7 + (i * 7)
            
            # Position: [X, Y, Z] + Quaternion: [W, X, Y, Z]
            # We add a small 0.01 gap from table and 0.001 between blocks for stability
            z_pos = table_surface_z + (block_half_height * 2 * i) + block_half_height + 0.01 + (i * 0.001)
            
            block_state = jp.array([block_curr_x, block_curr_y, z_pos, 1.0, 0.0, 0.0, 0.0])
            qpos = qpos.at[start_idx : start_idx + 7].set(block_state)
        # Initialize the physics pipeline
        data = self.pipeline_init(qpos, qvel)

        # 3. SETTLE THE BLOCKS: Step physics 50 times (~0.5 - 1.0s) with NO movement
        # We use a zero action (no torque/velocity change)
        settle_steps = 500
        
        def settle_fn(i, val):
            # Step physics with the current robot position as the constant target
            return self.pipeline_step(val, qpos[:7])
            
        # This runs entirely on the GPU during the reset call
        data = jax.lax.fori_loop(0, settle_steps, settle_fn, data)
        
        
        # --- Target and Metrics Logic ---
        rng, target_key = jax.random.split(rng)
        low, high = jp.array([0.2, -0.5, 0.3]), jp.array([0.8, 0.5, 1.0])
        random_target_pos = jax.random.uniform(target_key, (3,), minval=low, maxval=high)
        target_pos = self._fixed_target if self._fixed_target is not None else random_target_pos
        
        # Update mocap for target visualization
        data = data.replace(mocap_pos=data.mocap_pos.at[0].set(target_pos))

        reward, done = jp.zeros(2)
        metrics = {
            'top_displace': jp.zeros(()),
            'table_penalty': jp.zeros(()),
            'stability_penalty': jp.zeros(()),
            'dist_to_block': jp.zeros(()),
            'reward_top_knockdown': jp.zeros(()),
            'reward_bottom_knockdown': jp.zeros(()),
            'reward': jp.zeros(()),
            'success': jp.zeros(()),
            'reward_action_rate': jp.zeros(()),
        }

        info = {
            'prev_ctrl': jp.zeros(7),
            'prev_action': jp.zeros(7),
            'target_pos': target_pos,
            'block1_init_pos': data.qpos[7:10],  # Position of the first block in the stack
            'block2_init_pos': data.qpos[14:17],  # Position of the second block in the stack
            'block3_init_pos': data.qpos[21:24],  # Position of the third block in the stack
        }
        obs = self._get_obs(data, info)
                
        state = State(data, obs, reward, done, metrics, info)

        return state
            
    # todo: the init poses should be randomized
    def step(self, state: State, action: jp.ndarray) -> State:
        ctrl_delta = action * 0.1
        joint_state = state.pipeline_state.qpos[:7]
        ctrl = joint_state + ctrl_delta

        wrapped_ctrl = (ctrl + jp.pi) % (2 * jp.pi) - jp.pi
    
        # 4. Apply Hard Physical Limits to the Bounded Joints (1, 3, 5)
        # We define the limits for those specific indices
        low_lim = jp.array([-2.24, -2.57, -2.09])
        high_lim = jp.array([2.24, 2.57, 2.09])
        lim_idx = jp.array([1, 3, 5])
        
        # Clip only the limited joints to ensure they stay within MuJoCo's bounds
        final_ctrl = wrapped_ctrl.at[lim_idx].set(
            jp.clip(wrapped_ctrl[lim_idx], low_lim, high_lim)
        )
        
        # 1. Initial Height Constants
        table_z = self.sys.geom_pos[self._table_id, 2]
        block_height = 0.06 # 0.03 half-height * 2
        
        # 2. Step the physics
        data = self.pipeline_step(state.pipeline_state, final_ctrl)
        
        # 3. Extract Positions
        pos_0 = data.qpos[7:10]
        pos_1 = data.qpos[14:17]
        pos_2 = data.qpos[21:24]
        hand_pos = data.site_xpos[self._ee_site_idx]

        pos_0_init = state.info['block1_init_pos']
        pos_1_init = state.info['block2_init_pos']
        pos_2_init = state.info['block3_init_pos']

        
        # 4. Movement/Fall calculations
        # Stability check: Did the base blocks move?
        move_0 = jp.linalg.norm(pos_0[:2] - pos_0_init[:2])
        move_1 = jp.linalg.norm(pos_1[:2] - pos_1_init[:2])
        is_failure = (move_0 > 0.03) | (move_1 > 0.03)

        # If Z displacement is negative and larger than half the block height, it fell
        z_dropped = pos_2_init[2] - pos_2[2]
        is_success = z_dropped > 0.04 # Fell at least 8cm down
        # is_success = is_success & ~is_failure  # Success only if it fell and the bottom blocks are stable

        # table penetration check
        safety_margin = 0.05 
        penetration = table_z + safety_margin - hand_pos[2]
        # jax.debug.print("Hand Z: {hand_z:.3f} | Table Z: {table_z:.3f} | Penetration: {penetration:.3f}", hand_z=hand_pos[2], table_z=table_z, penetration=penetration)
        penalty_table = jp.where(penetration > 0, -10.0, 0.0)
        
        # 5. TERMINATION
        # done = jp.where(is_success | is_failure, 1.0, 0.0)
        done = jp.where(is_success | is_failure, 1.0, 0.0)
        # done = jp.zeros(())
        
        # 6. REWARD
        # Massive reward for the block actually falling
        top_reward_knockdown = jp.where(is_success, 80.0, 0.0)
        reward_bottom_knockdown = jp.where(is_failure, -100.0, 0.0)
        reward_knockdown = top_reward_knockdown + reward_bottom_knockdown
        
        # Heavy penalty for instability
        # penalty_stability = (jp.square(move_0) + jp.square(move_1)) * -5.0
        penalty_stability = 0.0

        penalty_action_rate = jp.linalg.norm(ctrl_delta) * -0.5
        
        # Dense guidance to get the hand to the block
        dist_to_top_block = jp.linalg.norm(hand_pos - pos_2)
        reward_approach = -dist_to_top_block
        
        reward = reward_knockdown + penalty_stability + reward_approach + penalty_table + penalty_action_rate
        
        # 7. Finalize
        metrics = {
            'top_displace': z_dropped,
            'table_penalty': penalty_table,
            'stability_penalty': penalty_stability,
            'dist_to_block': dist_to_top_block,
            'reward_top_knockdown': top_reward_knockdown,
            'reward_bottom_knockdown': reward_bottom_knockdown,
            'reward_action_rate': penalty_action_rate,
            'reward': reward,
            'success': jp.where(is_success & ~is_failure, 1.0, 0.0)
        }

        # update info
        info = state.info.copy()
        info['prev_ctrl'] = final_ctrl
        info['prev_action'] = action
        obs = self._get_obs(data, info)
        
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)        

    def _get_obs(self, data: mjx.Data, info: dict) -> jp.ndarray:
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

# Register the environment
envs.register_environment('cobot_reach', CobotEnv)

def progress_callback(num_steps, metrics, wandb_run):
    now = datetime.now().strftime('%H:%M:%S')
    
    # Use .get(key, default) to prevent KeyErrors
    reward = metrics.get('eval/episode_reward', 0.0)
    loss = metrics.get('training/total_loss', 0.0)
    sps = metrics.get('training/sps', 0.0)
    top_knockdown = metrics.get('eval/episode_reward_top_knockdown', 0.0)
    bot_knockdown = metrics.get('eval/episode_reward_bottom_knockdown', 0.0)
    top_displace = metrics.get('eval/episode_top_displace', 0.0)
    table_penalty = metrics.get('eval/episode_table_penalty', 0.0)
    dist_to_block = metrics.get('eval/episode_dist_to_block', 0.0)
    action_rate_penalty = metrics.get('eval/episode_reward_action_rate', 0.0)
    success = metrics.get('eval/episode_success', 0.0)

    print(f"[{now}] Steps: {num_steps:>10} | Reward: {reward:>10.2f} | Loss: {loss:>10.4f} | SPS: {sps:>8.0f} | Top KD: {top_knockdown:>6.2f} | Bot KD: {bot_knockdown:>6.2f} | Displace: {top_displace:>6.3f} | Table Penalty: {table_penalty:>6.3f} | Dist to Block: {dist_to_block:>6.3f}")
    wandb_run.log({
        'eval/episode_reward': reward,
        'training/total_loss': loss,
        'training/sps': sps,
        'eval/episode_reward_top_knockdown': top_knockdown,
        'eval/episode_reward_bottom_knockdown': bot_knockdown,
        'eval/episode_top_displace': top_displace,
        'eval/episode_table_penalty': table_penalty,
        'eval/episode_dist_to_block': dist_to_block,
        'eval/episode_reward_action_rate': action_rate_penalty,
        'eval/episode_success': success,
    })

def policy_video_callback(num_steps, make_inference_fn, params, wandb_run):
    eval_env = envs.get_environment('cobot_reach')
    jit_inference_fn = jax.jit(make_inference_fn(params))
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    pipeline_rollout = [state.pipeline_state]
    is_done_at = None # Track exactly when we finished
    
    n_steps = 500
    extra_steps = 50 # Render 1 second of extra footage after success

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        pipeline_rollout.append(state.pipeline_state)

        if state.done and is_done_at is None:
            is_done_at = i
            # Keep looping for extra_steps, but stop if we hit n_steps
            n_steps = min(i + extra_steps, n_steps) 

    # 1. Render the raw pixels
    # Brax returns a list of (H, W, 3) numpy arrays
    frames = eval_env.render(pipeline_rollout, camera="sideview")
    final_success = float(state.metrics['success'])
    color = (0, 255, 0) if final_success > 0.5 else (255, 0, 0)

    # 2. "Draw" the DONE indicator on the extra frames
    if is_done_at is not None:
        for f_idx in range(is_done_at, len(frames)):
            # Draw a simple red square in the top-left corner (50x50 pixels)
            # This is much faster than true text rendering in a headless environment
            frames[f_idx][10:60, 10:60, :] = color 

    # 3. Save and Log
    video_file = f"{video_path}/policy_video_{num_steps}.mp4"
    media.write_video(video_file, frames, fps=1.0 / eval_env.dt)
    
    wandb_run.log({
        'policy_video': wandb.Video(video_file, fps=30, format="mp4"),
    })

def domain_randomize(sys, rng):
    table_geom_idx  = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')

    @jax.vmap
    def rand(rng):
        # Randomize Table Height
        key1, key2 = jax.random.split(rng)
        new_z = jax.random.uniform(key1, (), minval=0.62, maxval=0.82)
        new_geom_pos = sys.geom_pos.at[table_geom_idx, 2].set(new_z)
        
        # Randomize Friction
        friction = jax.random.uniform(key2, (), minval=0.5, maxval=1.5)
        new_friction = sys.geom_friction.at[:, 0].set(friction)
        
        return new_geom_pos, new_friction

    batch_geom_pos, batch_friction = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_pos': 0,
        'geom_friction': 0,
    })
    sys = sys.tree_replace({
        'geom_pos': batch_geom_pos,
        'geom_friction': batch_friction,
    })

    return sys, in_axes

if __name__ == "__main__":

    # --- COMPILE ---
    ASSETS_DIR = "/home/luisamao/villa_spaces/sim_ws/src/mujoco_cobot/assets"
    ROBOT_XML = "/home/luisamao/villa_spaces/sim_ws/robot_simple_collision.xml"
    TABLE_XML = "/home/luisamao/villa_spaces/sim_ws/table.xml"
    xml_string = prepare_cobot_model(ROBOT_XML, TABLE_XML, ASSETS_DIR)
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    mj_data = mujoco.MjData(mj_model)

    print(f"Success! Model compiled with {mj_model.ngeom} geometries.")
    # renderer = mujoco.Renderer(mj_model)

    print(f"\nModel loaded successfully!")
    print(f"  - DOFs: {mj_model.nv}")
    print(f"  - Bodies: {mj_model.nbody}")
    print(f"  - Joints: {mj_model.njnt}")
    print(f"  - Actuators: {mj_model.nu}")

    # make partial functions for the training loop
    config = {
        "num_timesteps": 40_000_000,           # 40_000_000
        "num_evals": 20,
        "reward_scaling": 0.01,               # Lowered to stabilize Critic
        "episode_length": 500, # here
        "normalize_observations": True,
        "entropy_cost": 5e-3,
        "action_repeat": 1,
        "unroll_length": 20,                  # Increased for better GAE estimation
        "num_minibatches": 32,                # Increased for better gradient stochastics
        "num_updates_per_batch": 4,           # Lowered to prevent "over-correcting" on bad data
        "discounting": 0.99,                  # Lowered to focus the Critic's horizon
        "learning_rate": 3e-4,                # 3e-4
        "num_envs": 2048,                     # Moderate parallelization for stability
        "batch_size": 512,
        "seed": 42,
    }

    wandb_run = wandb.init(
        project="cobot-reach",
        config=config,
    )

    run_name = wandb_run.name
    video_path = f"videos/{run_name}"
    os.makedirs(video_path, exist_ok=True)

    train_fn = functools.partial(
        ppo.train,
        **config,
        # save_checkpoint_path = "playground_ppo_checkpoint",
    )
    custom_progress_callback = functools.partial(progress_callback, wandb_run=wandb_run)
    custom_policy_video_callback = functools.partial(policy_video_callback, wandb_run=wandb_run)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Training...")
    make_inference_fn, params, metrics = train_fn(environment=envs.get_environment('cobot_reach'), progress_fn=custom_progress_callback, policy_params_fn=custom_policy_video_callback, randomization_fn = domain_randomize)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Complete!")

    env_name = "cobot_reach"
    eval_env = envs.get_environment(env_name)
    model_path = 'mjx_knock_top_block' # wo knocking other blocks
    model_path = 'mjx_knock_top_block_no_penalty' # no penalty
    model_path = 'mjx_knock_top_block_no_penalty2' # no penalty. with stuff to make the loss not explode
    # model_path = 'mjx_ee_to_top_block' # trying to recreate target tracking
    model_path = 'mjx_knock_top_block_no_penalty_table_penalty_1' #
    model.save_params(model_path, params)
    print("saved params to", model_path)
    wandb.finish()