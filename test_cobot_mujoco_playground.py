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

def move_joint(env, target_joint_idx=0, max_steps=500):
    # 1. Setup JIT functions and Skeleton
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    num_envs = 4
    
    # Create a BATCH of keys (one for each env)
    rng = jax.random.PRNGKey(0)
    rng_batch = jax.random.split(rng, num_envs) 
    
    # Now reset will receive a (num_envs, 2) array instead of just (2,)
    init_state = jit_reset(rng_batch)
    
    # Create the hollow container to save memory during the rollout
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
        
        single_action = jnp.zeros(7)
        single_action = single_action.at[0].set(1.0) # Move first joint
        batched_action = jnp.broadcast_to(single_action, (4, 7))
        next_state = jit_step(state, batched_action)
        
        # Extract only the data MuJoCo needs to render
        m = next_state.metrics
        jax.debug.print(
            "--- STEP {step} (Env 0) ---\n"
            "Total Reward:  {rw:.4f} | Success: {succ}\n"
            "Dist/Block:    {dist:.4f} | Top Displace: {top_d:.4f}\n"
            "Knockdown (T): {kt:.2f} | Knockdown (B): {kb:.2f}\n"
            "Table Penalty: {tp:.2f} | Action Rate: {ar:.4f}\n",
            step=i,
            rw=next_state.reward[0],
            succ=m['success'][0],
            dist=m['reward/dist_to_block'][0],
            top_d=m['reward/top_displace'][0],
            kt=m['reward/reward_top_knockdown'][0],
            kb=m['reward/reward_bottom_knockdown'][0],
            tp=m['reward/table_penalty'][0],
            ar=m['reward/reward_action_rate'][0]
        )

        traj_data = empty_traj.tree_replace({
            "data.qpos": next_state.data.qpos,
            "data.qvel": next_state.data.qvel,
            "data.time": next_state.data.time,
            "data.ctrl": next_state.data.ctrl,
            "data.mocap_pos": next_state.data.mocap_pos,
            "data.mocap_quat": next_state.data.mocap_quat,
            "data.xfrc_applied": next_state.data.xfrc_applied,
        })
        
        # Debugging inside JIT if needed
        jax.debug.print("Step {i} qpos: {q}", i=i, q=next_state.data.qpos[target_joint_idx])
        
        return (next_state, og_pose), traj_data

    # 3. Execute Rollout on GPU
    og_pose = init_state.data.qpos # Initial joint positions
    _, trajectory = jax.lax.scan(
        step_fn, (init_state, og_pose), jnp.arange(max_steps)
    )

    # 4. Prepare for Renderer
    # Convert stacked JAX arrays into a Python list of individual states
    trajectory_world0 = jax.tree.map(lambda x: x[:, 0], trajectory)
    rollout_list = [
        jax.tree.map(lambda x, i=idx: jax.device_get(x[i]), trajectory_world0)
        for idx in range(max_steps)
    ]
    print(f"Shape of trajectory_world0 (qpos): {trajectory_world0.data.qpos.shape}")
    print(f"Length of rollout_list: {len(rollout_list)}")
    print(f"Shape of first item in rollout_list (qpos): {rollout_list[0].data.qpos.shape}")

    # 5. Render
    fps = 1.0 / env.dt
    frames = env.render(rollout_list, camera="sideview")
    print(f"Shape of frames (Video): {len(frames)}")
    media.write_video("test.mp4", frames, fps=fps)

# Usage
infer_env_cfg = default_config()
# infer_env_cfg = config_dict.ConfigDict(infer_env_cfg)
infer_env_cfg.vision_config.nworld = 4

# Initialize directly from the class as requested
infer_env = CobotEnv(config=infer_env_cfg)

# Wrap for Brax training compatibility (handles episode length and action repeat)
wrapped_infer_env = wrapper.wrap_for_brax_training(
    infer_env,
    episode_length=infer_env_cfg.episode_length, # or ppo_params.episode_length
    action_repeat=1,
)
move_joint(wrapped_infer_env, target_joint_idx=0)

# import jax
# import jax.numpy as jnp
# import mediapy as media

# def move_joint(env, num_envs=4, max_steps=500):
#     # 1. Setup Manually Vectorized JIT functions
#     # This replaces what the Brax wrapper was doing under the hood
#     jit_reset = jax.jit(jax.vmap(env.reset))
#     jit_step = jax.jit(jax.vmap(env.step))
    
#     # Create a BATCH of keys
#     rng = jax.random.PRNGKey(0)
#     rng_batch = jax.random.split(rng, num_envs) 
    
#     # Initialize state
#     init_state = jit_reset(rng_batch)
    
#     # Create the hollow container (same as before)
#     # We use init_state.data[0] to get the structure without the batch dim
#     empty_data = init_state.data.__class__(
#         **{k: None for k in init_state.data.__annotations__}
#     )
#     empty_traj = init_state.__class__(
#         **{k: None for k in init_state.__annotations__}
#     )
#     empty_traj = empty_traj.replace(data=empty_data)

#     # 2. Define the physics loop
#     def step_fn(carry, i):
#         state, og_pose = carry
        
#         # Define action (7 actuators)
#         single_action = jnp.zeros(7)
#         single_action = single_action.at[0].set(1.0) 
#         batched_action = jnp.broadcast_to(single_action, (num_envs, 7))
        
#         # Step the environment - NO AUTO RESET HAPPENS HERE
#         next_state = jit_step(state, batched_action)
        
#         # Debugging prints for Env 0
#         m = next_state.metrics
#         jax.debug.print(
#             "--- STEP {step} (Env 0) ---\n"
#             "Total Reward:  {rw:.4f} | Done: {done}\n"
#             "Dist/Block:    {dist:.4f} | Top Displace: {top_d:.4f}\n",
#             step=i,
#             rw=next_state.reward[0],
#             done=next_state.done[0], # You will see this stay 1.0 once it hits
#             dist=m['reward/dist_to_block'][0],
#             top_d=m['reward/top_displace'][0]
#         )

#         # Build trajectory data
#         traj_data = empty_traj.tree_replace({
#             "data.qpos": next_state.data.qpos,
#             "data.qvel": next_state.data.qvel,
#             "data.time": next_state.data.time,
#             "data.ctrl": next_state.data.ctrl,
#             "data.mocap_pos": next_state.data.mocap_pos,
#             "data.mocap_quat": next_state.data.mocap_quat,
#             "data.xfrc_applied": next_state.data.xfrc_applied,
#         })
        
#         return (next_state, og_pose), traj_data

#     # 3. Execute Rollout on GPU
#     og_pose = init_state.data.qpos 
#     _, trajectory = jax.lax.scan(
#         step_fn, (init_state, og_pose), jnp.arange(max_steps)
#     )

#     # 4. Squeeze to World 0 and move to CPU
#     trajectory_world0 = jax.tree.map(lambda x: x[:, 0], trajectory)
#     rollout_list = [
#         jax.tree.map(lambda x, i=idx: jax.device_get(x[i]), trajectory_world0)
#         for idx in range(max_steps)
#     ]

#     # 5. Render
#     fps = 1.0 / env.dt
#     frames = env.render(rollout_list, camera="sideview")
#     media.write_video("no_reset_test.mp4", frames, fps=fps)

# # --- EXECUTION ---
# infer_env_cfg = default_config()
# infer_env_cfg.vision_config.nworld = 4

# # Use the RAW environment class
# # This bypasses the Brax AutoResetWrapper entirely
# infer_env = CobotEnv(config=infer_env_cfg)

# move_joint(infer_env, num_envs=4, max_steps=100)