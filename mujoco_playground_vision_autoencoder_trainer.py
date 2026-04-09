import optax
from functools import partial
import jax
from cobot_networks import Generator
import jax.numpy as jnp
from utils import load_inference_without_env, get_obs_shape
from cobot_env import CobotEnv, default_config
import numpy as np
import wandb
import os
import orbax.checkpoint as ocp
from utils import denormalize
from dummy_logger import DummyWandbRun

# --- Configuration ---
learning_rate = 1e-4
batch_size = 8
num_iterations = 10 # 00
episode_length = 250

def collect_vision_data(env, teacher_policy, num_envs=32, episode_length=250, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    if len(key.shape) <= 1:
        jit_reset = jax.jit(jax.vmap(env.reset))
        jit_step = jax.jit(jax.vmap(env.step))
    else:
        # 1. Vectorize and JIT environment functions
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        # jit_reset = jax.jit(jax.vmap(env.reset))
        # jit_step = jax.jit(jax.vmap(env.step))
        
    # 2. Initialize environments
    reset_key, scan_key = jax.random.split(key)
    rng_batch = jax.random.split(reset_key, num_envs) 
    init_state = jit_reset(rng_batch)

    def step_fn(carry, _):
        state, rng = carry
        rng, policy_key = jax.random.split(rng)
        
        # 3. Teacher drives the robot to get "on-distribution" images
        # We assume teacher_policy returns (action, info)
        act, _ = teacher_policy(state.info["teacher_obs"], policy_key)
        
        # 4. Step environment
        next_state = jit_step(state, act)
        
        # 5. Extract and normalize pixels for the Autoencoder [0, 1]
        # We capture basecam and wristcam specifically
        pixels = {
            "basecam": next_state.obs['pixels/basecam'],
            "wristcam": next_state.obs['pixels/wristcam']
        }
        
        return (next_state, rng), pixels

    # 6. Run the scan loop (Entire rollout happens on GPU in one kernel)
    _, trajectory_pixels = jax.lax.scan(
        step_fn, (init_state, scan_key), None, length=episode_length
    )
    
    return trajectory_pixels

def train_autoencoder(teacher_policy, env, num_envs = 32, encoder_names=["basecam", "wristcam"], wandb_run = DummyWandbRun()):
    # 1. Setup Model and Params
    # We can use one model instance to init/apply for all cameras
    model = Generator()
    
    # Initialize a dictionary of params: {'basecam': params, 'wristcam': params}
    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
    params = {
        name: model.init(init_rngs, jnp.ones((1, 256, 256, 3)), train=True)['params'] 
        for name in encoder_names
    }
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params) # Optax handles the dict automatically

    @jax.jit
    def loss_fn(params, batch_pixels_dict, dropout_key):
        # batch_pixels_dict looks like {'basecam': array, 'wristcam': array}
        total_loss = 0.0
        reconstructions = {}
        
        for i, name in enumerate(encoder_names):
            rng = jax.random.fold_in(dropout_key, i) # Unique dropout per cam
            recon = model.apply({'params': params[name]}, batch_pixels_dict[name], 
                                train=True, rngs={'dropout': rng})
            total_loss += jnp.mean(jnp.square(recon - batch_pixels_dict[name]))
            reconstructions[name] = recon
            
        return total_loss / len(encoder_names), reconstructions

    @jax.jit
    def update_step(params, opt_state, batch_pixels_dict, dropout_key):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, recons), grads = grad_fn(params, batch_pixels_dict, dropout_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # --- Collection Loop ---
    rng = jax.random.PRNGKey(0)
    for i in range(num_iterations):
        rng, collect_key = jax.random.split(rng)
        traj_pixels = collect_vision_data(env, teacher_policy, num_envs=num_envs, key=collect_key)
        
        # 2. Prepare Data for all cameras
        # Map each camera to (Batch, H, W, C)
        batch_dict = {
            name: traj_pixels[name].reshape(-1, 256, 256, 3) 
            for name in encoder_names
        }
        
        # 3. Shuffle (using the same shuffle indices for both so temporal context matches)
        rng, shuffle_key = jax.random.split(rng)
        num_samples = next(iter(batch_dict.values())).shape[0]
        indices = jax.random.permutation(shuffle_key, jnp.arange(num_samples))
        batch_dict = {name: val[indices] for name, val in batch_dict.items()}
        
        epoch_losses = []
        for start_idx in range(0, num_samples, batch_size):
            mini_batch_dict = {name: val[start_idx:start_idx+batch_size] 
                               for name, val in batch_dict.items()}
            rng, dropout_key = jax.random.split(rng)
            
            params, opt_state, loss = update_step(
                params, opt_state, mini_batch_dict, dropout_key
            )
            epoch_losses.append(loss)

        if i % 2 == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"Iteration {i}, Multi-Cam Loss: {avg_loss:.6f}")
            
            # --- Visual Validation & Checkpointing ---
            log_data = {"autoencoder/total_loss": avg_loss, "iteration": i}
            
            for name in encoder_names:
                test_img = batch_dict[name][:4]
                recon = model.apply({'params': params[name]}, test_img, train=False)
                
                # Side-by-side comparison
                orig_cpu = denormalize(jax.device_get(test_img))
                recon_cpu = denormalize(jax.device_get(recon))
                comparison = np.concatenate([np.concatenate([orig_cpu, recon_cpu], axis=1)], axis=2)
                # Tile horizontally
                comparison = np.concatenate([img for img in comparison], axis=1) 
                
                log_data[f"autoencoder/{name}_recon"] = wandb.Image(comparison)

            wandb_run.log(log_data)

            # Save split params
            checkpointer = ocp.PyTreeCheckpointer()
            iter_path = os.path.abspath(f"checkpoints/autoencoder_run/iteration_{i}")
            ckpt = {
                'optimizer': opt_state,
                'iteration': i
            }
            for name in encoder_names:
                ckpt[f'{name}_params'] = params[name] # e.g. 'basecam_params': params['basecam']
            checkpointer.save(iter_path, ckpt)

        del batch_dict
        del traj_pixels
    return params

def debug_print_ckpt_structure(ckpt):
    print("\n" + "="*50)
    print("PRE-SAVE CHECKPOINT STRUCTURE DEBUG")
    print("="*50)
    
    def print_path_and_shape(path, val):
        # Convert the path (tuple of keys) into a readable string
        path_str = " -> ".join([str(p.key) if hasattr(p, 'key') else str(p) for p in path])
        
        # Check if it's a leaf (array) or something else
        if hasattr(val, 'shape'):
            print(f"PATH: {path_str} | SHAPE: {val.shape}")
        else:
            print(f"PATH: {path_str} | VALUE: {val} (Type: {type(val)})")

    # jax.tree_util.tree_map_with_path visits every single leaf in your dictionary
    jax.tree_util.tree_map_with_path(print_path_and_shape, ckpt)
    print("="*50 + "\n")

if __name__ == "__main__":
    env_cfg = default_config()
    num_envs = 16
    env_cfg['vision'] = True
    env_cfg['vision_config']['nworld'] = num_envs
    env_cfg['include_teacher_obs'] = True
    env = CobotEnv(config=env_cfg)

    restore_checkpoint_path = "/home/luisamao/villa_spaces/sim_ws/checkpoints/olive-bird-157/000045547520"
    teacher_obs_shape = get_obs_shape(env, num_envs, obs_key="teacher_obs")
    teacher_jit_inference_fn = load_inference_without_env(restore_checkpoint_path, obs_dim=teacher_obs_shape, action_dim= env.action_size)

    wandb_run = wandb.init(project="cobot_autoencoder", name="autoencoder_training_run")
    print("training autoencoder")
    train_autoencoder(teacher_jit_inference_fn, env, encoder_names = ["basecam", "wristcam"], wandb_run = wandb_run, num_envs = num_envs)
    print("done training autoencoder")