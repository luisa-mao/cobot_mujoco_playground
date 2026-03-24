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


from mujoco_playground._src import wrapper
from brax.training.agents.ppo import train as ppo
from cobot_env import CobotEnv, default_config
from ml_collections import config_dict
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import networks as ppo_networks
import functools
import copy
from datetime import datetime



# from mujoco_playground.config import dm_control_suite_params
# ppo_params = dm_control_suite_params.brax_vision_ppo_config(env_name)
# ppo_params


# 1. Load your env
env_cfg = default_config()
env_cfg['vision_config']['nworld'] = 4
print("num envs", env_cfg['vision_config']['nworld'])
env = CobotEnv(config=env_cfg)

# # 2. Setup Eval Env (often with a different number of worlds)
eval_env_cfg = copy.deepcopy(env_cfg)
eval_env_cfg['vision_config']['nworld'] = 4
print("eval_env_cfg num envs", eval_env_cfg['vision_config']['nworld'])
eval_env = CobotEnv(config=eval_env_cfg)

# network_factory = ppo_networks_vision.make_ppo_networks_vision
network_factory = ppo_networks.make_ppo_networks

num_envs = env_cfg['vision_config']['nworld']
eval_num_envs = eval_env_cfg['vision_config']['nworld']
print("here", num_envs, eval_num_envs)

def progress_callback(num_steps, metrics):
    print("progress function called")
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

import jax
import jax.numpy as jp
import mediapy as media
from brax.training.agents.ppo import train as ppo_utils
import mujoco

def policy_video_callback(num_steps, make_inference_fn, params):
    # 1. Setup Inference Environment (64 worlds for evaluation)
    # We use a deepcopy or a new ConfigDict to avoid mutating the training config
    infer_env_cfg = config_dict.ConfigDict(env_cfg)
    infer_env_cfg.vision_config.nworld = 64
    
    # Initialize directly from the class as requested
    infer_env = CobotEnv(config=infer_env_cfg)
    
    # Wrap for Brax training compatibility (handles episode length and action repeat)
    wrapped_infer_env = wrapper.wrap_for_brax_training(
        infer_env,
        episode_length=env_cfg.episode_length, # or ppo_params.episode_length
        action_repeat=1,
    )

    # 2. JIT the specific functions for this inference run
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    # 3. Setup Parallel Reset
    rng = jax.random.split(jax.random.PRNGKey(42), 64)
    reset_states = jax.jit(wrapped_infer_env.reset)(rng)

    # 4. Create the 'Skeleton' Trajectory (Hollow Container)
    # This prevents OOM by only storing what MuJoCo needs to render
    empty_data = reset_states.data.__class__(
        **{k: None for k in reset_states.data.__annotations__}
    )
    empty_traj = reset_states.__class__(
        **{k: None for k in reset_states.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data)

    # 5. Define the Step Function for lax.scan
    def step(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        act_keys = jax.random.split(act_key, 64)
        
        # Inference for all 64 envs
        act, _ = jit_inference_fn(state.obs, act_keys)
        state = wrapped_infer_env.step(state, act)
        
        # Selective data logging for rendering
        traj_data = empty_traj.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, rng), traj_data

    # 6. Execute Rollout (GPU)
    @jax.jit
    def do_rollout(state, rng):
        _, traj = jax.lax.scan(
            step, (state, rng), None, length=env_cfg.episode_length
        )
        return traj

    traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(0))
    
    # 7. Post-Process: Move axis from (Time, Batch) -> (Batch, Time)
    traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)

    # Pick the first environment (World 0) to render
    traj_world0 = jax.tree.map(lambda x: x[0], traj_stacked)
    
    # Convert the JAX Tensors back into a Python list of individual states
    rollout_list = [
        jax.tree.map(lambda x, i=i: jax.device_get(x[i]), traj_world0)
        for i in range(env_cfg.episode_length)
    ]

    # 8. Render and Save
    render_every = 2
    fps = 1.0 / infer_env.dt / render_every
    
    # MUJOCO VISUAL OPTIONS
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    frames = infer_env.render(
        rollout_list[::render_every], 
        height=480, width=640, 
        scene_option=scene_option
    )
    
    media.write_video(f"step_{num_steps}.mp4", frames, fps=fps)
    print(f"Logged video for step {num_steps}")


# # 3. Define the Training Function
train_fn = functools.partial(
    ppo.train,
    num_timesteps=1_000,
    num_evals=20,
    reward_scaling=0.01,
    episode_length=env_cfg.episode_length,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=5e-3,
    num_envs=num_envs,
    num_eval_envs=eval_num_envs,
    batch_size=4, #512
    network_factory=network_factory, # Use the CNN factory
    seed=42,
    max_devices_per_host = 1,
    progress_fn = progress_callback,
    policy_params_fn = policy_video_callback,
    clipping_epsilon=0.2,
    # clipping_epsilon_value=0.2,
    # use_pmap_on_reset = False
)

# # 4. RUN
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training, # Essential for MjxEnv
)