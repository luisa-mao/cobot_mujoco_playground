import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# os.environ["JAX_PLATFORMS"] = "cuda"

import jax
print(f"Total devices: {jax.device_count()}")
print(f"Local devices: {jax.local_device_count()}")

from cobot_env import CobotEnv, default_config
from ml_collections import config_dict
from mujoco_playground._src import wrapper


import jax
import jax.numpy as jnp
import mediapy as media

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint

def load_inference_without_env(model_path, obs_dim=30, action_dim=7):
    """
    Loads a pretrained Brax model using only known dimensions.
    """
    # 1. Manually create the observation 'blueprint'
    # This replaces env.reset(key).obs
    # obs_shape = jax.ShapeDtypeStruct(shape=(obs_dim,), dtype=jnp.dtype('float32'))

    # 2. Create the Networks
    # The factory needs the shape to build the internal MLP layers
    ppo_network = ppo_networks.make_ppo_networks(
        obs_dim, 
        action_dim, 
        preprocess_observations_fn=running_statistics.normalize
    )

    # 3. Make Inference Function (The "Shell")
    # This creates the logic wrapper for the policy
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # 4. Restore from Checkpoint
    # This pulls (normalizer_params, policy_params, value_params)
    params = checkpoint.load(model_path)
    
    # params[0]: Running mean/std (normalizer)
    # params[1]: Policy weights
    # params[2]: Value weights
    inference_params = (params[0], params[1], params[2])

    # 5. Create the JIT function
    raw_inference_fn = make_policy(inference_params)
    jit_inference_fn = jax.jit(raw_inference_fn)

    return jit_inference_fn

def move_joint(env, num_envs=4, max_steps=500):
    # 1. Setup Manually Vectorized JIT functions
    # This replaces what the Brax wrapper was doing under the hood
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    
    # Create a BATCH of keys
    rng = jax.random.PRNGKey(0)
    rng_batch = jax.random.split(rng, num_envs) 
    
    # Initialize state
    init_state = jit_reset(rng_batch)
    
    # Create the hollow container (same as before)
    # We use init_state.data[0] to get the structure without the batch dim
    empty_data = init_state.data.__class__(
        **{k: None for k in init_state.data.__annotations__}
    )
    empty_traj = init_state.__class__(
        **{k: None for k in init_state.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data)

    # 2. Define the physics loop
    def step_fn(carry, i):
        state, og_pose = carry
        num_joints = 8
        # Define action (num_joints actuators)
        single_action = jnp.zeros(num_joints)
        # single_action = single_action.at[0].set(1.0) 
        single_action = single_action.at[7].set(1.0) 
        batched_action = jnp.broadcast_to(single_action, (num_envs, num_joints))
        
        # Step the environment - NO AUTO RESET HAPPENS HERE
        next_state = jit_step(state, batched_action)
        
        # Debugging prints for Env 0
        m = next_state.metrics
        jax.debug.print(
            "--- STEP {step} (Env 0) ---\n"
            "Total Reward:  {rw:.4f} | Done: {done}\n"
            "Dist/Block:    {dist:.4f} | Top Displace: {top_d:.4f}\n",
            step=i,
            rw=next_state.reward[0],
            done=next_state.done[0], # You will see this stay 1.0 once it hits
            dist=m['reward/dist_to_block'][0],
            top_d=m['reward/top_displace'][0]
        )
        jax.debug.print(
            "--- STEP {step} ---\n"
            "Robot QPos: {robot_q}\n",
            # "Block 1 QPos: {b1_q}\n",
            step=i,
            robot_q=next_state.data.qpos[0, num_joints-2:num_joints],      # Arm + Gripper
            # b1_q=next_state.data.qpos[0, num_joints:num_joints+7] # First Block
        )

        # Build trajectory data
        traj_data = empty_traj.tree_replace({
            "data.qpos": next_state.data.qpos,
            "data.qvel": next_state.data.qvel,
            "data.time": next_state.data.time,
            "data.ctrl": next_state.data.ctrl,
            "data.mocap_pos": next_state.data.mocap_pos,
            "data.mocap_quat": next_state.data.mocap_quat,
            "data.xfrc_applied": next_state.data.xfrc_applied,
        })
        
        return (next_state, og_pose), traj_data

    # 3. Execute Rollout on GPU
    og_pose = init_state.data.qpos 
    _, trajectory = jax.lax.scan(
        step_fn, (init_state, og_pose), jnp.arange(max_steps)
    )

    # 4. Squeeze to World 0 and move to CPU
    trajectory_world0 = jax.tree.map(lambda x: x[:, 0], trajectory)
    rollout_list = [
        jax.tree.map(lambda x, i=idx: jax.device_get(x[i]), trajectory_world0)
        for idx in range(max_steps)
    ]

    # 5. Render
    fps = 1.0 / env.dt
    frames = env.render(rollout_list, camera="topdown")
    media.write_video("topdown_vis.mp4", frames, fps=fps)
    frames = env.render(rollout_list, camera="sideview")
    media.write_video("sideview_vis.mp4", frames, fps=fps)
    frames = env.render(rollout_list, camera="sideview_y")
    media.write_video("sideview_y_vis.mp4", frames, fps=fps)

from brax.io import model
from brax.training.agents.ppo import train as ppo
import functools
from brax.training.agents.ppo import networks as ppo_networks

def examine_policy(env, num_envs=4, max_steps=500, restore_checkpoint_path=''):
    training_config = {
        "num_timesteps": 0, # 100_000, # 40_000_000,
        "num_evals": 1, # 2_000, # here
        "reward_scaling": 0.01,
        "episode_length": 250,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 10,
        "num_minibatches": 32,
        "num_updates_per_batch": 4,
        "discounting": 0.99,
        "learning_rate": 3e-4,
        "entropy_cost": 5e-3,
        "num_envs": num_envs,
        "batch_size": 512,
        "network_factory": ppo_networks.make_ppo_networks,
        "seed": 42,
        "max_devices_per_host": 1,
        "clipping_epsilon": 0.2,
        "restore_checkpoint_path": restore_checkpoint_path,
    }
    train_fn = functools.partial(
        ppo.train,
        **training_config,
    )
    # make_inference_fn, params, metrics = train_fn(environment=env, num_timesteps=0, wrap_env_fn=wrapper.wrap_for_brax_training) # dummy train to get the make_inference_fn
    # inference_fn = make_inference_fn(params)
    # inference_fn = jax.jit(make_inference_fn(params))
    # jit_inference_fn = jax.jit(inference_fn)
    jit_inference_fn = load_inference_without_env(restore_checkpoint_path, obs_dim=env.observation_size, action_dim=env.action_size)

    # 1. Setup Manually Vectorized JIT functions
    # This replaces what the Brax wrapper was doing under the hood
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    
    # Create a BATCH of keys
    rng = jax.random.PRNGKey(0)
    rng_batch = jax.random.split(rng, num_envs) 
    
    # Initialize state
    init_state = jit_reset(rng_batch)
    
    # Create the hollow container (same as before)
    # We use init_state.data[0] to get the structure without the batch dim
    empty_data = init_state.data.__class__(
        **{k: None for k in init_state.data.__annotations__}
    )
    empty_traj = init_state.__class__(
        **{k: None for k in init_state.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data)

    # 2. Define the physics loop
    def step_fn(carry, i):
        state, rng = carry
        
        act_rng, rng = jax.random.split(rng)
        state_keys = list(vars(state).keys())
        jax.debug.print("state attributes {z}", z=state_keys)
        jax.debug.print("obs shape {z}", z = state.obs.shape)
        jax.debug.print("obs {z}", z = state.obs[0])
        jax.debug.print("data shape {z}", z = state.data.shape)
        # obs shape (Array(4, dtype=int32), Array(33, dtype=int32))
        # state attributes ['data', 'obs', 'reward', 'done', 'metrics', 'info']
        # data shape (Array(4, dtype=int32), Array(15, dtype=int32), Array(6, dtype=int32))
        ctrl, _ = jit_inference_fn(state.obs, act_rng)        
        next_state = jit_step(state, ctrl)
        jax.debug.print("control {z}", z = ctrl[0])
        
        # Debugging prints for Env 0
        m = next_state.metrics
        jax.debug.print(
            "--- STEP {step} (Env 0) ---\n"
            "Total Reward:  {rw:.4f} | Done: {done}\n"
            "Dist/Block:    {dist:.4f} | Top Displace: {top_d:.4f}\n",
            step=i,
            rw=next_state.reward[0],
            done=next_state.done[0], # You will see this stay 1.0 once it hits
            dist=m['reward/dist_to_block'][0],
            top_d=m['reward/top_displace'][0]
        )

        # Build trajectory data
        traj_data = empty_traj.tree_replace({
            "data.qpos": next_state.data.qpos,
            "data.qvel": next_state.data.qvel,
            "data.time": next_state.data.time,
            "data.ctrl": next_state.data.ctrl,
            "data.mocap_pos": next_state.data.mocap_pos,
            "data.mocap_quat": next_state.data.mocap_quat,
            "data.xfrc_applied": next_state.data.xfrc_applied,
        })
        
        return (next_state, rng), traj_data

    # 3. Execute Rollout on GPU
    # og_pose = init_state.data.qpos 
    _, trajectory = jax.lax.scan(
        step_fn, (init_state, rng), jnp.arange(max_steps)
    )

    # 4. Squeeze to World 0 and move to CPU
    trajectory_world0 = jax.tree.map(lambda x: x[:, 0], trajectory)
    rollout_list = [
        jax.tree.map(lambda x, i=idx: jax.device_get(x[i]), trajectory_world0)
        for idx in range(max_steps)
    ]

    # 5. Render
    fps = 1.0 / env.dt
    frames = env.render(rollout_list, camera="sideview")
    media.write_video("examine_policy.mp4", frames, fps=fps)


# --- EXECUTION ---
infer_env_cfg = default_config()
infer_env_cfg.vision_config.nworld = 4
infer_env_cfg.num_blocks = 3

# Use the RAW environment class
# This bypasses the Brax AutoResetWrapper entirely
infer_env = CobotEnv(config=infer_env_cfg)

move_joint(infer_env, num_envs=4, max_steps=100)
# restore_checkpoint_path = "/home/luisamao/villa_spaces/sim_ws/checkpoints/dry-tree-91/000043089920"
# examine_policy(infer_env, num_envs=4, max_steps=100, restore_checkpoint_path=restore_checkpoint_path)