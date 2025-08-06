import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

deltas = jnp.array([
    [0, 1],   # up
    [0, -1],  # down
    [-1, 0],  # left
    [1, 0],   # right
])  # shape (4, 2)


def roll_top_left(arr):
    # Roll up until first row is not all zeros
    def cond_row(x):
        return jnp.all(x[0] == 0)

    def body_row(x):
        return jnp.roll(x, shift=-1, axis=0)

    arr = jax.lax.while_loop(cond_row, body_row, arr)

    # Roll left until first column is not all zeros
    def cond_col(x):
        return jnp.all(x[:, 0] == 0)

    def body_col(x):
        return jnp.roll(x, shift=-1, axis=1)

    arr = jax.lax.while_loop(cond_col, body_col, arr)

    return arr


@jax.jit
def sample_coord(key, prob_matrix):
    # Flatten the 4x4 matrix
    probs_flat = prob_matrix.reshape(-1)
    path_grid = jnp.full_like(prob_matrix, 0.0)

    # Sample an index according to the probability distribution
    idx = jr.choice(key, a=probs_flat.shape[0], p=probs_flat)

    # Convert flat index back to (row, col)
    row = idx // prob_matrix.shape[1]
    col = idx % prob_matrix.shape[1]

    prob_matrix = prob_matrix.at[row, col].set(
        False)  # Set the sampled position to 0
    # Mark the sampled position in the path grid
    path_grid = path_grid.at[row, col].set(1.0)

    return jnp.array([row, col]), prob_matrix, path_grid


@eqx.filter_jit
def create_puzzle(
        key: PRNGKeyArray,
        grid_size: int = 4,
        n_pieces: int = 4,
        min_piece_size: int = 2,
        max_piece_size: int = 5):

    key_walk, key_sizes = jr.split(key)
    init_grid = jnp.ones((grid_size, grid_size), dtype=bool)
    piece_sizes = jr.randint(key_sizes, shape=(
        n_pieces,), minval=min_piece_size, maxval=max_piece_size+1)

    def place_piece(carry, piece_size):
        key, visited_grid = carry
        new_key, _ = jr.split(key)

        path_init = jnp.full((max_piece_size, 2), -1)
        start_pos, visited_grid, path_grid = sample_coord(key, visited_grid)

        path_init = path_init.at[0].set(start_pos)

        def step_fn(carry, _):
            pos, visited, path, path_grid, step, done, key, piece_size = carry
            key, subkey = jr.split(key)

            too_long = step >= piece_size - 1

            def early_exit():
                return (pos, visited, path, path_grid, step, True,
                        key, piece_size), None

            def do_step():
                candidates = pos + deltas  # (4, 2)
                in_bounds = jnp.all((candidates >= 0) & (
                    candidates < grid_size), axis=1)
                cx, cy = candidates.T
                not_visited = visited[cx, cy]
                valid_mask = in_bounds & not_visited

                # If no valid moves, done=True
                any_valid = jnp.any(valid_mask)

                def no_valid():
                    return (pos, visited, path, path_grid, step, True,
                            key, piece_size), None

                def has_valid():
                    # Assign large negative logits to invalid moves so
                    # they won't be sampled
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    move_idx = jr.categorical(key, logits)
                    new_pos = candidates[move_idx]
                    visited_updated = visited.at[new_pos[0], new_pos[1]].set(
                        False)
                    path_grid_updated = path_grid.at[new_pos[0],
                                                     new_pos[1]].set(
                        1.0)
                    path_updated = path.at[step + 1].set(new_pos)
                    return (new_pos, visited_updated, path_updated,
                            path_grid_updated, step + 1, False, key,
                            piece_size), None

                return jax.lax.cond(any_valid, has_valid, no_valid)

            return jax.lax.cond(done | too_long, early_exit, do_step)

        init_state = (start_pos, visited_grid, path_init,
                      path_grid, 0, False, key, piece_size)

        final_state, _ = jax.lax.scan(
            step_fn,
            init=init_state,
            xs=None,
            length=max_piece_size - 1)
        _, final_grid, _, new_path_grid, _, _, _, _ = final_state

        return (new_key, final_grid), new_path_grid
    init_carry = (key_walk, init_grid)
    (_, final_state), pieces = jax.lax.scan(
        place_piece,
        init=init_carry,
        xs=piece_sizes,
    )

    return jnp.concat((final_state[None, :], pieces), axis=0)
