from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Any


from region_generator import random_walk

deltas = jnp.array([
    [0, 1],   # up
    [0, -1],  # down
    [-1, 0],  # left
    [1, 0],   # right
])  # shape (4, 2)


class EnvParams(eqx.Module):
    grid_size: int = 4
    max_steps: int = 100


class EnvState(eqx.Module):
    grid: jax.Array
    pieces: jax.Array


class PackingGameEnv():

    @partial(jax.jit, static_argnames=("self", "params",))
    def reset(self, key: KeyArray, params: EnvParams):
        full_path = random_walk(
            key=key,
            max_steps=params.max_steps,
            grid_size=params.grid_size
        )

        state = EnvState(
            grid=...,
            pieces=...,
        )
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
