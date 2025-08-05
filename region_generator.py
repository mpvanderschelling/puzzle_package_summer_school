import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray


@eqx.filter_jit
def random_walk(key: PRNGKeyArray, max_steps=100, grid_size=10):
    deltas = jnp.array([
        [0, 1],   # up
        [0, -1],  # down
        [-1, 0],  # left
        [1, 0],   # right
    ])  # shape (4, 2)

    key_start, key_scan, key_chunks = jr.split(key, 3)
    start_pos = jr.randint(key_start, (2,), 0, grid_size)

    visited_init = jnp.zeros((grid_size, grid_size), dtype=bool)
    visited_init = visited_init.at[start_pos[0], start_pos[1]].set(True)

    path_init = jnp.full((max_steps + 1, 2), -1)
    path_init = path_init.at[0].set(start_pos)

    def step_fn(carry, _):
        pos, visited, path, step, done, key = carry
        key, subkey = jr.split(key)

        def early_exit():
            return (pos, visited, path, step, True, key), None

        def do_step():
            candidates = pos + deltas  # (4, 2)
            in_bounds = jnp.all((candidates >= 0) & (
                candidates < grid_size), axis=1)
            cx, cy = candidates.T
            not_visited = ~visited[cx, cy]
            valid_mask = in_bounds & not_visited

            # If no valid moves, done=True
            any_valid = jnp.any(valid_mask)

            def no_valid():
                return (pos, visited, path, step, True, key), None

            def has_valid():
                # Assign large negative logits to invalid moves so they won't be sampled
                logits = jnp.where(valid_mask, 0.0, -1e9)
                move_idx = jr.categorical(key, logits)
                new_pos = candidates[move_idx]
                visited_updated = visited.at[new_pos[0], new_pos[1]].set(True)
                path_updated = path.at[step + 1].set(new_pos)
                return (new_pos, visited_updated, path_updated, step + 1, False, key), None

            return jax.lax.cond(any_valid, has_valid, no_valid)

        return jax.lax.cond(done, early_exit, do_step)

    init_state = (start_pos, visited_init, path_init, 0, False, key)

    final_state, _ = jax.lax.scan(
        step_fn,
        init=init_state,
        xs=None,
        length=max_steps)
    _, _, full_path, _, _, _ = final_state

    chunks = random_chunks(
        key_chunks, (jnp.sum(full_path != -1) // 2), max_steps+1)

    return full_path, chunks.cumsum()


allowed_sizes = jnp.arange(start=2, stop=5)


@eqx.filter_jit
def random_chunks(key: PRNGKeyArray, n: jax.Array, max_steps: int):
    def cond_fun(state):
        remaining, *_ = state
        return remaining > 0

    def body_fun(state):
        remaining, key, length, chunks = state
        key, subkey = jax.random.split(key)

        # If emraining is exactly in allowed_sizes, use it and finish
        is_in_remaining_sizes = jnp.any(allowed_sizes == remaining)

        # if remaining is in allowed_sizes or less than min allowed size, take it all
        # this prevents infinite loops
        is_terminal = jnp.logical_or(
            is_in_remaining_sizes, remaining < allowed_sizes.min())

        def take_terminal():
            return remaining

        def take_random():
            return jax.random.choice(subkey, allowed_sizes)

        next_chunk = jax.lax.cond(is_terminal, take_terminal, take_random)

        chunks = chunks.at[length].set(next_chunk)
        length += 1
        remaining -= next_chunk

        return (remaining, key, length, chunks)

    chunks = jnp.zeros(max_steps, dtype=jnp.int32)
    init_state = (n, key, 0, chunks)

    remaining, key, length, chunks = jax.lax.while_loop(
        cond_fun, body_fun, init_state)
    return chunks


def roll_top_left(arr):
    # Roll up until first row is not all zeros
    def cond_row(x): return jnp.all(x[0] == 0)
    def body_row(x): return jnp.roll(x, shift=-1, axis=0)

    arr = jax.lax.while_loop(cond_row, body_row, arr)

    # Roll left until first column is not all zeros
    def cond_col(x): return jnp.all(x[:, 0] == 0)
    def body_col(x): return jnp.roll(x, shift=-1, axis=1)

    arr = jax.lax.while_loop(cond_col, body_col, arr)

    return arr
