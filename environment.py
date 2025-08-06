from __future__ import annotations

from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import PRNGKeyArray

from plotting_fn import COLORMAPS
from region_generator import create_puzzle, roll_top_left


class EnvParams(eqx.Module):
    grid_size: int = 4
    n_pieces: int = 4
    min_piece_size: int = 2
    max_piece_size: int = 4


class EnvState(eqx.Module):
    grid: jax.Array  # shape (pieces+1, grid_size, grid_size), dtype=bool

    @classmethod
    def init(cls, key: PRNGKeyArray, params: EnvParams):
        grid = create_puzzle(
            key,
            grid_size=params.grid_size,
            n_pieces=params.n_pieces,
            min_piece_size=params.min_piece_size,
            max_piece_size=params.max_piece_size,
        )
        return cls(grid=grid)

    def roll_top_left(self) -> EnvState:
        rolled_grid = jnp.concatenate(
            [self.grid[None, 0],
             jax.vmap(roll_top_left)(self.grid[1:]),
             ]
        )
        return EnvState(grid=rolled_grid)

    def plot(self):
        fig, axes = plt.subplots(1, 5, figsize=(5, 1))
        for i, ax in enumerate(axes):
            ax.imshow(self.grid[i], cmap=COLORMAPS[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])


class PackingGameEnv():

    @partial(jax.jit, static_argnames=("self", "params",))
    def reset(self, key: PRNGKeyArray, params: EnvParams):
        state = EnvState.init(key, params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        obs = jnp.concatenate(
            [state.grid[None, 0],
             jax.vmap(roll_top_left)(state.grid[1:]),
             ]
        )

        return obs.astype(jnp.float32)

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        ...
