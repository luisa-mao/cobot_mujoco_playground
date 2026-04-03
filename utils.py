from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
import jax
import jax.numpy as jp
import functools
from brax.training import distribution
from brax.training import networks
from flax import linen

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


def create_student_network(obs_dim, action_dim):
    # 1. Use the factory to create the architecture
    # This includes the preprocess_observations_fn (Standardizer)
    ppo_nets = ppo_networks.make_ppo_networks(
        observation_size=obs_dim,
        action_size=action_dim,
        preprocess_observations_fn=running_statistics.normalize
    )
    # parametric_action_distribution = distribution.NormalTanhDistribution(
    #     event_size=action_dim
    # )
    # student_policy = networks.make_policy_network(
    #   parametric_action_distribution.param_size,
    #   obs_dim,
    #   preprocess_observations_fn=running_statistics.normalize,
    #   hidden_layer_sizes=(32,) * 4,
    #   activation=linen.swish,
    #   obs_key = 'state',
    #   distribution_type='tanh_normal',
    #   noise_std_type='scalar',
    #   init_noise_std=1.0,
    #   state_dependent_std=False,
    #   kernel_init=jax.nn.initializers.lecun_uniform(),
    #   mean_clip_scale=None,
    # )

    # 2. Randomly Initialize Parameters
    rng = jax.random.PRNGKey(0)
    p_key, n_key = jax.random.split(rng)
    
    # Initialize the normalizer (Preprocessing)
    initial_norm_params = running_statistics.init_state(
        jax.ShapeDtypeStruct(shape=(obs_dim,), dtype=jp.float32)
    )

    # Initialize the policy network (MLP)
    # initial_policy_params = student_policy.init(p_key)
    initial_policy_params = ppo_nets.policy_network.init(p_key)

    # We return the pieces needed for both Training and Inference
    return {
        # "policy_network": student_policy,
        # "dist": parametric_action_distribution,
        "policy_network": ppo_nets.policy_network,
        "dist": ppo_nets.parametric_action_distribution,
        "params": (initial_norm_params, initial_policy_params)
    }

def get_student_inference_fn(student_dict):
    policy_net = student_dict["policy_network"]
    dist = student_dict["dist"]

    def inference_fn(params, obs, key):
        # 1. Preprocessing & MLP
        # apply() handles the internal preprocess_observations_fn
        logits = policy_net.apply(params[0], params[1], obs)
        
        # 2. Post-processing. this needs to be deterministic,
        # since that's how it's trained in the distillation
        postprocessed_actions = dist.mode(logits)
        
        return postprocessed_actions, {}
    return inference_fn

def get_video(jit_inference_fn, infer_env, num_envs = 4, episode_length=250, rng_seed = 0, render_every = 2):
    jit_reset = jax.jit(jax.vmap(infer_env.reset))
    jit_step = jax.jit(jax.vmap(infer_env.step))
    
    rng = jax.random.PRNGKey(rng_seed)
    rng_batch = jax.random.split(rng, num_envs) 
    reset_states = jit_reset(rng_batch)
    
    # Setup Skeletons
    empty_data = reset_states.data.__class__(
        **{k: None for k in reset_states.data.__annotations__}
    )
    empty_traj = reset_states.__class__(
        **{k: None for k in reset_states.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data, metrics={})

    # Define the Step Function
    def step_fn(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
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
            metrics={'success': state.metrics['success'], 'blocks_fell': state.metrics['blocks_fell']},
            done=state.done
        )
        return (state, rng), traj_data

    # Execute Rollout
    _, traj_stacked = jax.lax.scan(
        step_fn, (reset_states, rng), None, length=episode_length
    )

    # Post-Process (World 0)
    traj_world0 = jax.tree.map(lambda x: x[:, 0], traj_stacked)
    
    # Move signals to CPU for logic processing
    success_signal = jax.device_get(traj_world0.metrics['success'])
    blocks_fell_signal = jax.device_get(traj_world0.metrics['blocks_fell'])
    
    # Find the FIRST index where 'done' is true
    done_indices = jp.where(blocks_fell_signal > 0.5)[0]
    is_done_at = int(done_indices[0]) if len(done_indices) > 0 else None

    # Convert JAX Tensors to list for MuJoCo renderer
    rollout_list = [
        jax.tree.map(lambda x, i=i: jax.device_get(x[i]), traj_world0)
        for i in range(episode_length)
    ]

    # Render
    render_every = 2
    
    frames = infer_env.render(
        rollout_list[::render_every], 
        height=480, width=640, 
        camera="sideview"
    )
    
    # Dynamic Success Marker
    # Determine the color based on the final frame's success metric
    final_is_success = success_signal[-1] > 0.5
    marker_color = [0, 255, 0] if final_is_success else [255, 0, 0]

    if is_done_at is not None:
        # Convert the physics index to the rendered frames index
        render_done_idx = is_done_at // render_every
        for f_idx in range(render_done_idx, len(frames)):
            # Draw the square only from the moment the task was 'done'
            is_success = success_signal[f_idx * render_every] > 0.5
            marker_color = [0, 255, 0] if is_success else [255, 0, 0]
            frames[f_idx][10:60, 10:60, :] = marker_color
    return frames, final_is_success
