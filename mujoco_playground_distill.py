import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags



from utils import load_inference_without_env
import jax

import jax
import jax.numpy as jp
from cobot_env import CobotEnv, default_config
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import networks as ppo_networks
from utils import create_student_network, get_student_inference_fn, get_video, create_student_network_vision
from brax.training.acme import running_statistics
import optax
import mediapy as media
import functools
import wandb
import argparse

class DummyWandbRun:
    def __init__(self) -> None:
        self.name = "dummy_run"
    def log(self, data):
        pass
    def finish(self):
        pass


def dagger_rollout(env, student_policy, teacher_policy, beta=0.5, num_envs=4, max_steps=500, key=None, wandb_run=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Vectorize the environment functions
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    
    # Initialize batch of environments
    reset_key, scan_key = jax.random.split(key)
    rng_batch = jax.random.split(reset_key, num_envs) 
    init_state = jit_reset(rng_batch)

    def step_fn(carry, i):
        state, rng = carry
        
        # 1. Split keys for stochasticity
        rng, teacher_key, student_key, mix_key = jax.random.split(rng, 4)
        mix_keys = jax.random.split(mix_key, num_envs)
        
        # 2. Get actions from BOTH (Teacher is our 'Label', Student is our 'Actor')
        teacher_act, _ = teacher_policy(state.info["teacher_obs"], teacher_key)
        student_act, _ = student_policy(state.obs, student_key)
        
        # 3. Beta-selection: Decide which action to actually EXECUTE
        # DAgger often runs the student but learns from the teacher's response
        use_teacher = jax.random.bernoulli(mix_key, p=beta, shape=(num_envs, 1))
        executed_act = jp.where(use_teacher, teacher_act, student_act)
        
        # 4. Step environment
        next_state = jit_step(state, executed_act)
        
        # 5. Build Trajectory Package
        # We store the OBS and the TEACHER'S ACTION (the target for learning)
        traj_step = {
            "obs": state.obs,
            "expert_action": teacher_act, 
            "executed_action": executed_act,
            "reward": next_state.reward,
            "done": next_state.done,

            # for rendering
            # "data": next_state.data
        }
        
        return (next_state, rng), traj_step

    # Run the scan loop
    (final_state, _), trajectory = jax.lax.scan(
        step_fn, (init_state, scan_key), jp.arange(max_steps)
    )
    total_reward_env0 = jp.sum(trajectory['reward'][:, 0])
    # Move to CPU to print
    print(f"Total Reward (Env 0): {jax.device_get(total_reward_env0):.2f}")
    if wandb_run is not None:
        wandb_run.log({'distill/total_reward_env0': jax.device_get(total_reward_env0)})

    rewards_per_env = jp.sum(trajectory['reward'], axis=0)
    mean_batch_reward = jp.mean(rewards_per_env)

    if wandb_run is not None:
        wandb_run.log({'distill/total_reward_env0': jax.device_get(total_reward_env0)})
        wandb_run.log({'distill/mean_batch_reward': jax.device_get(mean_batch_reward)})

    # trajectory_world0 = jax.tree.map(lambda x: x[:, 0], trajectory)
    # # 2. Get a single state object to use as a "skeleton"
    # # We take the first environment's initial state (index 0)
    # state_skeleton = jax.tree.map(lambda x: x[0], init_state)

    # # 3. Reconstruct the list of State objects
    # rollout_list = []
    # for idx in range(max_steps):
    #     # We take the skeleton and replace its internal data with step 'idx'
    #     # jax.tree.map slices every leaf in the PyTree at that time step
    #     step_data = jax.tree.map(lambda x: jax.device_get(x[idx]), trajectory_world0)
        
    #     # Merge into the skeleton. 
    #     # This works because MujocoPlayground states are usually named tuples or dataclasses
    #     step_state = state_skeleton.replace(
    #         data=step_data['data'],
    #         obs=step_data['obs'],
    #         reward=step_data['reward'],
    #         done=step_data['done']
    #     )
    #     rollout_list.append(step_state)

    # # Render
    # render_every = 2
    # fps = 1.0 / env.dt / render_every
    
    # frames = env.render(
    #     rollout_list[::render_every], 
    #     height=480, width=640, 
    #     camera="sideview"
    # )
    # media.write_video("distill_debug/dagger_rollout.mp4", frames, fps=fps)
    return trajectory


def loss_fn(student_params, normalizer_params, obs, expert_actions):

    # def preprocess_pixels(k, v):
    #     if isinstance(k, str) and "pixels" in k and v.dtype == jp.uint8:
    #         return v.astype(jp.float32) / 255.0
    #     return v

    # # Use tree_util.tree_map_with_path to check keys
    # obs = jax.tree_util.tree_map_with_path(
    #     lambda path, val: preprocess_pixels(jax.tree_util.keystr(path), val), 
    #     obs
    # )

    logits = student["policy_network"].apply(normalizer_params, student_params, obs)
    predicted_actions = student["dist"].mode(logits)
    return jp.mean(jp.square(predicted_actions - expert_actions))

# Standard JAX/Optax update step
@jax.jit
def update_step(student_params, opt_state, obs, expert_actions):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    normalizer_params, policy_params = student_params

    # if obs is a pytree
    norm_obs = {"joint_states": obs["joint_states"]} if isinstance(obs, dict) else obs # if using vision
    new_norm_params = running_statistics.update(
        normalizer_params, 
        norm_obs
    )
    loss, grads = grad_fn(policy_params, normalizer_params, obs, expert_actions)
    updates, opt_state = optimizer.update(grads, opt_state, params=policy_params)
    policy_params = optax.apply_updates(policy_params, updates)
    return (new_norm_params, policy_params), opt_state, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cobot with MJX')
    parser.add_argument('--vision', action='store_true', help='Enable vision-based training')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_notes', type=str, default="", help='Override wandb notes')
    args = parser.parse_args()
    vision = args.vision

    # main training loop
    env_cfg = default_config()
    num_envs = 32
    env_cfg['vision'] = vision
    env_cfg['vision_config']['nworld'] = num_envs
    env_cfg['include_teacher_obs'] = True
    env = CobotEnv(config=env_cfg)

    reset_key = jax.random.PRNGKey(0)
    rng_batch = jax.random.split(reset_key, num_envs) 
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(rng_batch)

    jax.tree_util.tree_map(lambda x: print(f"Array: {x.shape}"), env_state.obs)

    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.obs)

    print("\n--- Network Input Shapes (The 'Blueprint') ---")
    print(obs_shape)

    teacher_obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.info["teacher_obs"])
    print("\n--- Teacher Obs Shapes (The 'Labels') ---")
    print(teacher_obs_shape)

    obs_dim = obs_shape
    action_dim = env.action_size
    NUM_DAGGER_ITERATIONS = 100
    STEPS_PER_ITERATION = 250 * num_envs
    BETA_DECAY = 0.95  # Decay teacher influence over time

    # make student
    if vision:
        student = create_student_network_vision(obs_dim=obs_dim, action_dim=action_dim) # returns dict of (policy_network, dist, params)
    else:
        student = create_student_network(obs_dim=obs_dim, action_dim=action_dim) # returns dict of (policy_network, dist, params)
    student_params = student["params"]

    # teacher
    restore_checkpoint_path = "/home/luisamao/villa_spaces/sim_ws/checkpoints/glad-paper-140/000045547520"
    teacher_jit_inference_fn = load_inference_without_env(restore_checkpoint_path, obs_dim=teacher_obs_shape, action_dim=action_dim)
    student_jit_inference_fn = get_student_inference_fn(student)


    # optimizer
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(student["params"][1])
    rng = jax.random.PRNGKey(42)

    # wandb
    wandb_run = DummyWandbRun()
    if args.wandb:
        wandb_run = wandb.init(
            project="cobot-distill",
            notes=args.wandb_notes,
            # config=wandb_config, # Changed from training_config to wandb_config
        )

    for iteration in range(NUM_DAGGER_ITERATIONS):
        # Calculate current beta (prob of using teacher actions)
        
        student_jit_inference_fn_w_params = functools.partial(student_jit_inference_fn, student_params)
        current_beta = BETA_DECAY ** iteration
        
        # A. Collect Dataset via Rollout
        rng, rollout_key = jax.random.split(rng)
        trajectory = dagger_rollout(
            env, 
            student_jit_inference_fn_w_params, 
            teacher_jit_inference_fn, 
            beta=current_beta, 
            num_envs=num_envs, 
            max_steps=STEPS_PER_ITERATION // num_envs, 
            key=rollout_key,
            wandb_run=wandb_run,
        )

        render_every = 2

        if iteration % 5 == 0:
            fps = 1.0 / env.dt / render_every
            frames, final_is_success = get_video(student_jit_inference_fn_w_params, env, num_envs=num_envs, episode_length=250, rng_seed=iteration, render_every = render_every)
            video_file = f"distill_debug_{wandb_run.name}/it_{iteration}_distill.mp4"
            os.makedirs(os.path.dirname(video_file), exist_ok=True)
            media.write_video(video_file, frames, fps=fps)
            frames, final_is_success_teacher = get_video(teacher_jit_inference_fn, env, num_envs=num_envs, episode_length=250, rng_seed=iteration, render_every = render_every, obs_key="teacher_obs")
            teacher_video_file = f"distill_debug_{wandb_run.name}/teacher_it_{iteration}_teacher.mp4"
            media.write_video(f"distill_debug_{wandb_run.name}/teacher_it_{iteration}_teacher.mp4", frames, fps=fps)
            wandb_run.log({
                'policy_video': wandb.Video(video_file, format="mp4"),
            })
            wandb_run.log({
                'teacher_video': wandb.Video(teacher_video_file, format="mp4"),
            })
        
        # B. Flatten the trajectory for training
        # trajectory['obs'] shape: [steps, num_envs, obs_dim] -> [total_samples, obs_dim]
        # obs_batch = trajectory['obs'].reshape(-1, obs_dim)

        # we use -1 to squash (steps * envs) and then keep the rest of the dimensions
        obs_batch = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), 
            trajectory['obs']
)        
        expert_act_batch = trajectory['expert_action'].reshape(-1, action_dim)

        # C. Supervised Learning Update
        # In a real DAgger setup, you'd usually append this to a growing 'Buffer'
        # but here we update on the fresh batch

        total_samples = expert_act_batch.shape[0] # using minibatches to prevent oom
        batch_size = 16
        for epoch in range(40):  # Multiple gradient steps per rollout

            rng, subkey = jax.random.split(rng)
            perm = jax.random.permutation(subkey, total_samples)
            for i in range(0, total_samples, batch_size):
                indices = perm[i : i + batch_size]
                mini_obs = jax.tree.map(lambda x: x[indices], obs_batch)
                mini_act = expert_act_batch[indices]
                student_params, opt_state, loss = update_step(
                    student_params, opt_state, mini_obs, mini_act
                )
        del obs_batch
        del expert_act_batch
        del trajectory
        
        print(f"Iteration {iteration} | Beta: {current_beta:.2f} | Loss: {loss:.6f}")
        wandb_run.log({
            'distill/loss': loss,
            'distill/beta': current_beta,
            'distill/final_success': final_is_success,
        })

    wandb_run.finish()