from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from region_generator import create_puzzle

deltas = jnp.array([
    [0, 1],   # up
    [0, -1],  # down
    [-1, 0],  # left
    [1, 0],   # right
])  # shape (4, 2)


class EnvParams(eqx.Module):
    grid_size: int = 4
    n_pieces: int = 4
    min_piece_size: int = 2
    max_piece_size: int = 4


class EnvState(eqx.Module):
    grid: jax.Array  # shape (pieces+1, grid_size, grid_size), dtype=bool


class PackingGameEnv():

    @partial(jax.jit, static_argnames=("self", "params",))
    def reset(self, key: PRNGKeyArray, params: EnvParams):
        puzzle_fn = partial(
            create_puzzle,
            grid_size=params.grid_size,
            n_pieces=params.n_pieces,
            min_piece_size=params.min_piece_size,
            max_piece_size=params.max_piece_size,
        )

        state = puzzle_fn(key)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        return state.grid.astype(jnp.float32)

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        ...
