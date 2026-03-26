from typing import Tuple
from cobot_env import CobotEnv

import jax
import mujoco
from mujoco import mjx

def dummy_domain_randomize(sys):
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    return sys, in_axes

# i think add block pos randomization here instead of reset
def domain_randomize(mjx_model: mjx.Model, num_worlds: int) -> Tuple[mjx.Model, mjx.Model]:
    mj_model = CobotEnv().mj_model
    table_geom_idx  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
    rng = rng = jax.random.PRNGKey(0)
    @jax.vmap
    def rand(rng):
        # Randomize Table Height
        key1, key2 = jax.random.split(rng)
        new_z = jax.random.uniform(key1, (), minval=0.6, maxval=0.7)
        new_geom_pos = mjx_model.geom_pos.at[table_geom_idx, 2].set(new_z)
        
        # Randomize Friction
        friction = jax.random.uniform(key2, (), minval=0.5, maxval=1.5)
        new_friction = mjx_model.geom_friction.at[:, 0].set(friction)
        
        return new_geom_pos, new_friction

    batch_geom_pos, batch_friction = rand(jax.random.split(rng, num_worlds))
    in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({
        'geom_pos': 0,
        'geom_friction': 0,
    })
    mjx_model = mjx_model.tree_replace({
        'geom_pos': batch_geom_pos,
        'geom_friction': batch_friction,
    })
    jax.debug.print("geom pos {z} shape {s}", z = batch_geom_pos, s = batch_geom_pos.shape)
    jax.debug.print("table heights {z}", z = batch_geom_pos[:, table_geom_idx, 2])
    # jax.debug.print("domain randomize called")
    return mjx_model, in_axes


def domain_randomize_batch(mjx_model: mjx.Model, rng) -> Tuple[mjx.Model, mjx.Model]: # expects batch of rng
    mj_model = CobotEnv().mj_model
    table_geom_idx  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
    rng = rng = jax.random.PRNGKey(0)
    @jax.vmap
    def rand(rng):
        # Randomize Table Height
        key1, key2 = jax.random.split(rng)
        new_z = jax.random.uniform(key1, (), minval=0.52, maxval=0.72)
        new_geom_pos = mjx_model.geom_pos.at[table_geom_idx, 2].set(new_z)
        
        # Randomize Friction
        friction = jax.random.uniform(key2, (), minval=0.5, maxval=1.5)
        new_friction = mjx_model.geom_friction.at[:, 0].set(friction)
        
        return new_geom_pos, new_friction

    # batch_geom_pos, batch_friction = rand(jax.random.split(rng, num_worlds))
    batch_geom_pos, batch_friction = rand(jax.random.split(rng))
    in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({
        'geom_pos': 0,
        'geom_friction': 0,
    })
    mjx_model = mjx_model.tree_replace({
        'geom_pos': batch_geom_pos,
        'geom_friction': batch_friction,
    })
    # jax.debug.print("geom pos {z} shape {s}", z = batch_geom_pos, s = batch_geom_pos.shape)
    # jax.debug.print("table heights {z}", z = batch_geom_pos[:, table_geom_idx, 2])
    jax.debug.print("domain randomize called")
    return mjx_model, in_axes
