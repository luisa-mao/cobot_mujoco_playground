import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from typing import Tuple

from cobot_env import CobotEnv, default_config
from brax.training.agents.ppo import networks as ppo_networks
import copy
import functools
from mujoco_playground._src import wrapper
import mediapy as media
import numpy as np

import jax

from domain_randomization import dummy_domain_randomize, domain_randomize


# 1. Load your env
env_cfg = default_config()
env_cfg['vision_config']['nworld'] = 64
print("num envs", env_cfg['vision_config']['nworld'])
env = CobotEnv(config=env_cfg)

# network_factory = ppo_networks_vision.make_ppo_networks_vision
network_factory = ppo_networks.make_ppo_networks
domain_randomize_64 = functools.partial(domain_randomize, num_worlds = 64)
wrapped_env = wrapper.wrap_for_brax_training(
    env,
    episode_length=200,
    action_repeat=1,
    randomization_fn=domain_randomize_64,
)

jit_reset = jax.jit(wrapped_env.reset)
jit_step = jax.jit(wrapped_env.step)

def tile(img, d):
    assert img.shape[0] == d*d
    img = img.reshape((d,d)+img.shape[1:])
    return np.concat(np.concat(img, axis=1), axis=1)

state = jit_reset(jax.random.split(jax.random.PRNGKey(0), 64))
image_name = "domain_randomize.png"
media.write_image(image_name, tile(state.info['frame_stack'][:64], 8), width=512)
print("saved to", image_name)
# # 1. Select the first 64 environments from your 64-batch state
# # We need to slice the data and the metrics to avoid passing 64 envs to the renderer
# indices = jnp.arange(64)
# state_64 = jax.tree.map(lambda x: x[indices], state)
# batched_mjx_model, in_axes = domain_randomize(env.mjx_model, num_worlds=64)
# rendered_list = []
# def slice_model(batched_model, in_axes, i):
#     def _slice(ax, leaf):
#         # If ax is 0, we slice the batch dimension
#         if ax is not None:
#             return leaf[i]
#         # If ax is None, we return the original leaf (constant across envs)
#         return leaf

#     # CRITICAL: treat None as a leaf so JAX doesn't crash on structural mismatch
#     return jax.tree.map(
#         _slice, 
#         in_axes, 
#         batched_model, 
#         is_leaf=lambda x: x is None
#     )

# for i in range(64):
#     # Slice the state and the specific model for this environment index
#     single_state = jax.tree.map(lambda x: x[i], state_64)
#     single_mjx_model = slice_model(batched_mjx_model, in_axes, i)
    
#     # STEP A: Forward Kinematics
#     # This ensures single_state.data.geom_xpos is updated for physics
#     new_data = mjx.forward(single_mjx_model, single_state.data)
#     single_state = single_state.replace(data=new_data)
    
#     # STEP B: Update the Renderer's C-Model
#     # env.render uses env.mj_model. We must copy our randomized JAX values 
#     # into this C-struct so the pixels match the physics.
#     print(single_mjx_model.geom_pos.shape)
#     print(env.mj_model.geom_pos.shape)
#     env.mj_model.geom_pos[18] = single_mjx_model.geom_pos[18]
#     print("single state qpos", single_state.data.qpos)
#     print("table height", single_mjx_model.geom_pos[18])
    
#     # STEP C: Render
#     # The renderer now uses the updated env.mj_model and the forwarded single_state.data
#     img = env.render( # this is using mj model which doesnt have the domain randomization. maybe need to render inside the env to see it. do this when figuring out vision based thing
#         single_state, 
#         width=256, 
#         height=256, 
#         camera="sideview"
#     )
#     rendered_list.append(img)

# # 3. Stack them into a numpy array (64, 256, 256, 3)
# rendered_frames = np.array(rendered_list)

# # 4. Tile and show
# tiled_image = tile(rendered_frames, 8)

# media.write_image("domain_randomize.png", tiled_image, width=1024)