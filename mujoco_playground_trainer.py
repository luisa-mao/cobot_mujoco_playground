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
from domain_randomization import domain_randomize_batch

import wandb


# from mujoco_playground.config import dm_control_suite_params
# ppo_params = dm_control_suite_params.brax_vision_ppo_config(env_name)
# ppo_params


# 1. Load your env
env_cfg = default_config()
env_cfg['vision_config']['nworld'] = 1024
print("num envs", env_cfg['vision_config']['nworld'])
env = CobotEnv(config=env_cfg)

# # 2. Setup Eval Env (often with a different number of worlds)
eval_env_cfg = copy.deepcopy(env_cfg)
eval_env_cfg['vision_config']['nworld'] = 64 # 256
print("eval_env_cfg num envs", eval_env_cfg['vision_config']['nworld'])
eval_env = CobotEnv(config=eval_env_cfg)

# network_factory = ppo_networks_vision.make_ppo_networks_vision
network_factory = ppo_networks.make_ppo_networks

num_envs = env_cfg['vision_config']['nworld']
eval_num_envs = eval_env_cfg['vision_config']['nworld']
print("here", num_envs, eval_num_envs)

def progress_callback(num_steps, metrics, wandb_run):
    now = datetime.now().strftime('%H:%M:%S')
    
    # Standard PPO metrics
    reward = metrics.get('eval/episode_reward/reward', 0.0)
    loss = metrics.get('training/total_loss', 0.0)
    sps = metrics.get('training/sps', 0.0)
    
    # Custom environment metrics
    # Note the "reward/" prefix is part of your original key name
    top_knockdown = metrics.get('eval/episode_reward/reward_top_knockdown', 0.0)
    bot_knockdown = metrics.get('eval/episode_reward/reward_bottom_knockdown', 0.0)
    top_displace  = metrics.get('eval/episode_reward/top_displace', 0.0)
    table_penalty = metrics.get('eval/episode_reward/table_penalty', 0.0)
    dist_to_block = metrics.get('eval/episode_reward/dist_to_block', 0.0)
    action_rate   = metrics.get('eval/episode_reward/reward_action_rate', 0.0)
    success       = metrics.get('eval/episode_success', 0.0)
    print(f"[{now}] Steps: {num_steps:>10} | Reward: {reward:>10.2f} | Loss: {loss:>10.4f} | SPS: {sps:>8.0f} | Top KD: {top_knockdown:>6.2f} | Bot KD: {bot_knockdown:>6.2f} | Displace: {top_displace:>6.3f} | Table Penalty: {table_penalty:>6.3f} | Dist to Block: {dist_to_block:>6.3f} | Action rate: {action_rate:>6.3f} | Success: {success:>6.3f}")

    train_reward = metrics.get('training/episode_reward/reward', 0.0)
    train_success = metrics.get('training/episode_success', 0.0)

    train_top_knockdown = metrics.get('training/episode_reward/reward_top_knockdown', 0.0)
    train_bot_knockdown = metrics.get('training/episode_reward/reward_bottom_knockdown', 0.0)
    train_top_displace  = metrics.get('training/episode_reward/top_displace', 0.0)
    train_table_penalty = metrics.get('training/episode_reward/table_penalty', 0.0)
    train_dist_to_block = metrics.get('training/episode_reward/dist_to_block', 0.0)
    train_action_rate   = metrics.get('training/episode_reward/reward_action_rate', 0.0)

    print(f"[{now}] Train Steps: {num_steps:>10} | Reward: {train_reward:>10.2f} | Loss: {loss:>10.4f} | SPS: {sps:>8.0f} | Top KD: {train_top_knockdown:>6.2f} | Bot KD: {train_bot_knockdown:>6.2f} | Displace: {train_top_displace:>6.3f} | Table Penalty: {train_table_penalty:>6.3f} | Dist to Block: {train_dist_to_block:>6.3f} | Action rate: {train_action_rate:>6.3f} | Success: {train_success:>6.3f}")

    wandb_run.log({
        'eval/episode_reward': reward,
        'training/total_loss': loss,
        'training/sps': sps,
        'eval/episode_reward_top_knockdown': top_knockdown,
        'eval/episode_reward_bottom_knockdown': bot_knockdown,
        'eval/episode_top_displace': top_displace,
        'eval/episode_table_penalty': table_penalty,
        'eval/episode_dist_to_block': dist_to_block,
        'eval/episode_reward_action_rate': action_rate,
        'eval/episode_success': success,
    })


import jax
import jax.numpy as jp
import mediapy as media
from brax.training.agents.ppo import train as ppo_utils
import mujoco

def policy_video_callback(num_steps, make_inference_fn, params, wandb_run, num_envs = 4
):
    infer_env_cfg = default_config()
    infer_env_cfg.vision_config.nworld = num_envs
    episode_length = infer_env_cfg.episode_length

    # Use the RAW environment class
    infer_env = CobotEnv(config=infer_env_cfg)
    # wrapped_infer_env = wrapper.wrap_for_brax_training(
    #     infer_env,
    #     episode_length=episode_length,
    #     action_repeat=1,
    # )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=False)) # here
    jit_reset = jax.jit(jax.vmap(infer_env.reset))
    jit_step = jax.jit(jax.vmap(infer_env.step))
    
    rng = jax.random.PRNGKey(num_steps)
    rng_batch = jax.random.split(rng, num_envs) 
    reset_states = jit_reset(rng_batch)
    
    # 4. Setup Skeletons
    empty_data = reset_states.data.__class__(
        **{k: None for k in reset_states.data.__annotations__}
    )
    empty_traj = reset_states.__class__(
        **{k: None for k in reset_states.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data, metrics={})

    # 5. Define the Step Function
    def step_fn(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        # act_keys = jax.random.split(act_key, num_envs)
        
        act, _ = jit_inference_fn(state.obs, act_key)
        state = jit_step(state, act)
        
        traj_data = empty_traj.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        # Track both success and the overall done flag
        traj_data = traj_data.replace(
            metrics={'success': state.metrics['success']},
            done=state.done
        )
        return (state, rng), traj_data

    # 6. Execute Rollout
    _, traj_stacked = jax.lax.scan(
        step_fn, (reset_states, rng), None, length=episode_length
    )

    # 7. Post-Process (World 0)
    traj_world0 = jax.tree.map(lambda x: x[:, 0], traj_stacked)
    
    # Move signals to CPU for logic processing
    success_signal = jax.device_get(traj_world0.metrics['success'])
    done_signal = jax.device_get(traj_world0.done)
    
    # Find the FIRST index where 'done' is true
    done_indices = jp.where(done_signal > 0.5)[0]
    is_done_at = int(done_indices[0]) if len(done_indices) > 0 else None

    # Convert JAX Tensors to list for MuJoCo renderer
    rollout_list = [
        jax.tree.map(lambda x, i=i: jax.device_get(x[i]), traj_world0)
        for i in range(episode_length)
    ]

    # 8. Render
    render_every = 2
    fps = 1.0 / infer_env.dt / render_every
    
    frames = infer_env.render(
        rollout_list[::render_every], 
        height=480, width=640, 
        camera="sideview"
    )
    
    # 9. Dynamic Success Marker
    # Determine the color based on the final frame's success metric
    final_is_success = success_signal[-1] > 0.5
    marker_color = [0, 255, 0] if final_is_success else [255, 0, 0]

    if is_done_at is not None:
        # Convert the physics index to the rendered frames index
        render_done_idx = is_done_at // render_every
        for f_idx in range(render_done_idx, len(frames)):
            # Draw the square only from the moment the task was 'done'
            frames[f_idx][10:60, 10:60, :] = marker_color

    video_path = f"videos/{wandb_run.name}"
    os.makedirs(video_path, exist_ok=True)
    video_file = f"{video_path}/step_{num_steps}.mp4"
    media.write_video(video_file, frames, fps=fps)
    print(f"Logged video for step {num_steps}. Success: {final_is_success}")
    wandb_run.log({
        'policy_video': wandb.Video(video_file, fps=30, format="mp4"),
    })
    
# # 3. Define the Training Function
# Create the training configuration
training_config = {
    "num_timesteps": 100_000, # 40_000_000,
    "num_evals": 2_000, # here
    "reward_scaling": 0.01,
    "episode_length": env_cfg.episode_length,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 10,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.99,
    "learning_rate": 6e-5,
    "entropy_cost": 5e-3,
    "num_envs": num_envs,
    "num_eval_envs": eval_num_envs,
    "batch_size": 512,
    "network_factory": network_factory,
    "seed": 42,
    "max_devices_per_host": 1,
    "clipping_epsilon": 0.2,
}

wandb_run = wandb.init(
    project="cobot-reach-mujoco-playground",
    config=training_config,
)

wandb_progress_callback = functools.partial(progress_callback, wandb_run=wandb_run)
wandb_policy_video_callback = functools.partial(policy_video_callback, wandb_run=wandb_run, num_envs = training_config['num_eval_envs'])


train_fn = functools.partial(
    ppo.train,
    progress_fn = wandb_progress_callback,
    policy_params_fn = wandb_policy_video_callback,
    save_checkpoint_path = f"/home/luisamao/villa_spaces/sim_ws/checkpoints/{wandb_run.name}",
    **training_config,
)

# # 4. RUN
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    randomization_fn = domain_randomize_batch,
    wrap_env_fn=wrapper.wrap_for_brax_training, # Essential for MjxEnv
)

# save
# from brax.io import model

# os.makedirs("checkpoints", exist_ok=True)
# model_path = f"checkpoints/{wandb_run.name}"
# model.save_params(model_path, params)
# print("saved params to", model_path)
wandb.finish()

# todo: domain randomization. then try with vision