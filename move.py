import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def padded_translate(arr, shift, grid_size: int = 4):
    # Step 1: Pad array with zeros
    padded = jnp.pad(
        arr, ((0, grid_size-1), (0, grid_size-1)), mode='constant')

    # Step 2: Roll the padded array
    rolled = jnp.roll(padded, shift=shift, axis=(0, 1))
    rolled_cropped = rolled[:grid_size, :grid_size]
    off_grid_penalty = rolled.sum() - rolled_cropped.sum()
    # Step 3: Crop the central 4x4 region
    return rolled[:grid_size, :grid_size], off_grid_penalty
