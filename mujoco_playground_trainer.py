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
from utils import get_video

import jax
import jax.numpy as jp
import mediapy as media
# from brax.training.agents.ppo import train as ppo_utils
import argparse

import wandb

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
    xy_displace   = metrics.get('eval/episode_reward/xy_displace', 0.0)
    table_penalty = metrics.get('eval/episode_reward/table_penalty', 0.0)
    reward_approach = metrics.get('eval/episode_reward/reward_approach', 0.0)
    action_rate   = metrics.get('eval/episode_reward/reward_action_rate', 0.0)
    action_direction   = metrics.get('eval/episode_reward/reward_action_direction', 0.0)
    success       = metrics.get('eval/episode_success', 0.0)
    success_rate       = metrics.get('eval/episode_success_rate', 0.0)
    print(f"[{now}] Steps: {num_steps:>10} | Reward: {reward:>10.2f} | Loss: {loss:>10.4f} | SPS: {sps:>8.0f} | Top KD: {top_knockdown:>6.2f} | Bot KD: {bot_knockdown:>6.2f} | Displace: {top_displace:>6.3f} | \
          XY Displace: {xy_displace:>6.3f} | Table Penalty: {table_penalty:>6.3f} | Reward Approach: {reward_approach:>6.3f} | \
             Action direction: {action_direction:>6.3f} | Action rate: {action_rate:>6.3f} | \
            Success: {success:>6.3f} | Success Rate: {success_rate:>6.3f} | ")

    wandb_run.log({
        'eval/episode_reward': reward,
        'training/total_loss': loss,
        'training/sps': sps,
        'eval/episode_reward_top_knockdown': top_knockdown,
        'eval/episode_reward_bottom_knockdown': bot_knockdown,
        'eval/episode_top_displace': top_displace,
        'eval/episode_xy_displace': xy_displace,
        'eval/episode_table_penalty': table_penalty,
        'eval/episode_reward_approach': reward_approach,
        'eval/episode_reward_action_rate': action_rate,
        'eval/episode_reward_action_direction': action_direction,
        'eval/episode_success': success,
        'eval/episode_success_rate': success_rate,
    })


def policy_video_callback(num_steps, make_inference_fn, params, wandb_run, infer_env, num_envs = 4, episode_length=250):

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=False)) # here
    render_every = 2
    frames, final_is_success = get_video(jit_inference_fn, infer_env, num_envs=num_envs, episode_length=episode_length, rng_seed = num_steps, render_every=render_every)
    video_path = f"videos/{wandb_run.name}"
    os.makedirs(video_path, exist_ok=True)
    video_file = f"{video_path}/step_{num_steps}.mp4"
    fps = 1.0 / infer_env.dt / render_every
    media.write_video(video_file, frames, fps=fps)
    print(f"Logged video for step {num_steps}. Success: {final_is_success}")
    wandb_run.log({
        'policy_video': wandb.Video(video_file, fps=fps, format="mp4"),
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cobot with MJX')
    parser.add_argument('--vision', action='store_true', help='Enable vision-based training')
    parser.add_argument('--num_envs', type=int, default=None, help='Override number of envs')
    parser.add_argument('--wandb_notes', type=str, default="", help='Override wandb notes')
    args = parser.parse_args()

    env_cfg = default_config()
    eval_env_cfg = copy.deepcopy(env_cfg)
    
    vision = args.vision
    n_world = args.num_envs if args.num_envs else 1024
    n_world_eval = 64
    normalize_observations = True
    num_minibatches = 32
    batch_size = 512
    network_factory = ppo_networks.make_ppo_networks

    if vision:
        n_world = args.num_envs if args.num_envs else 256
        n_world_eval = 16
        normalize_observations = False
        num_minibatches = 4
        batch_size = 64
        network_factory = functools.partial(
            ppo_networks_vision.make_ppo_networks_vision,
            policy_obs_key="joint_states",
            value_obs_key="joint_states",
        )

    env_cfg['vision'] = vision
    env_cfg['vision_config']['nworld'] = n_world
    env = CobotEnv(config=env_cfg)

    eval_env_cfg['vision'] = vision
    eval_env_cfg['vision_config']['nworld'] = n_world_eval 
    eval_env = CobotEnv(config=eval_env_cfg)

    print(f"Mode: {'VISION' if vision else 'STATE'} | Envs: {n_world}")
    
    num_envs = env_cfg['vision_config']['nworld']
    eval_num_envs = eval_env_cfg['vision_config']['nworld']
    print("Num envs:", num_envs, eval_num_envs)

    training_config = {
        "num_timesteps": 100_000, # 40_000_000,
        "num_evals": 2_000, # here
        "reward_scaling": 0.01,
        "episode_length": env_cfg.episode_length,
        "normalize_observations": normalize_observations,
        "action_repeat": 1,
        "unroll_length": 10,
        "num_minibatches": num_minibatches,
        "num_updates_per_batch": 4,
        "discounting": 0.99,
        "learning_rate": 3e-4,
        "entropy_cost": 5e-3,
        "num_envs": num_envs,
        "num_eval_envs": eval_num_envs,
        "batch_size": batch_size,
        "network_factory": network_factory,
        "seed": 42,
        "max_devices_per_host": 1,
        "clipping_epsilon": 0.2,
    }

    # Ensure env_cfg is converted to a dict correctly
    env_cfg_dict = dict(env_cfg)

    # Use the union operator (|) to merge dictionaries
    wandb_config = training_config | env_cfg_dict

    # Initialize wandb using the merged config
    wandb_run = wandb.init(
        project="cobot-reach-mujoco-playground",
        notes=args.wandb_notes,
        config=wandb_config, # Changed from training_config to wandb_config
    )

    wandb_progress_callback = functools.partial(progress_callback, wandb_run=wandb_run)
    wandb_policy_video_callback = functools.partial(policy_video_callback, wandb_run=wandb_run, num_envs = training_config['num_eval_envs'], infer_env = eval_env, episode_length=eval_env_cfg.episode_length)

    train_fn = functools.partial(
        ppo.train,
        progress_fn = wandb_progress_callback,
        policy_params_fn = wandb_policy_video_callback,
        save_checkpoint_path = f"/home/luisamao/villa_spaces/sim_ws/checkpoints/{wandb_run.name}",
        **training_config,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        randomization_fn = domain_randomize_batch,
        wrap_env_fn=wrapper.wrap_for_brax_training, # Essential for MjxEnv
    )

    wandb.finish()
