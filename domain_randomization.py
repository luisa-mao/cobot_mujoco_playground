from typing import Tuple, Optional
from cobot_env import CobotEnv

import jax
import mujoco
from mujoco import mjx
import jax.numpy as jp

def dummy_domain_randomize(sys):
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    return sys, in_axes

# i think add block pos randomization here instead of reset
def domain_randomize(mjx_model: mjx.Model,
                        rng: Optional[jax.Array] = None,
                        num_worlds: Optional[int] = None,
                        num_blocks: int = 3) -> Tuple[mjx.Model, mjx.Model]:
    if (rng is None) == (num_worlds is None):
        raise ValueError("Must provide exactly one of `rng` or `num_worlds`")
    mj_model = CobotEnv().mj_model
    table_geom_idx  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
    block_geom_idxs = jp.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f'geom_block_{i}') for i in range(num_blocks)])
    print("block geom idx", block_geom_idxs)

    @jax.vmap
    def rand(rng):
        key_pos, key_fric, key_color = jax.random.split(rng, 3)

        # 1. Randomize Table Height
        new_z = jax.random.uniform(key_pos, (), minval=0.70, maxval=0.75) # actual table height is 29 inches
        new_geom_pos = mjx_model.geom_pos.at[table_geom_idx, 2].set(new_z)

        # 2. Randomize Friction
        friction = jax.random.uniform(key_fric, (), minval=0.5, maxval=1.5)
        new_friction = mjx_model.geom_friction.at[:, 0].set(friction)

        # 3. Randomize Color (Jitter)
        # Base: 0.60 0.55 0.50. We jitter only the RGB (3 channels)
        base_color = jp.array([0.60, 0.55, 0.50])
        jitter = jax.random.uniform(key_color, (num_blocks, 3), minval=-0.02, maxval=0.02)
        new_rgb = jp.clip(base_color + jitter, 0.0, 1.0)
        
        # Apply the new RGB to the specific block indices in the rgba array
        # Note: we keep alpha (index 3) as 1.0
        new_rgba = mjx_model.geom_rgba.at[block_geom_idxs, :3].set(new_rgb)
        
        return new_geom_pos, new_friction, new_rgba

    if rng is None and num_worlds is not None:
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, num_worlds)
    batch_geom_pos, batch_friction, batch_rgba = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({
        'geom_pos': 0,
        'geom_friction': 0,
        'geom_rgba': 0,
    })

    mjx_model = mjx_model.tree_replace({
        'geom_pos': batch_geom_pos,
        'geom_friction': batch_friction,
        'geom_rgba': batch_rgba,
    })
    return mjx_model, in_axes