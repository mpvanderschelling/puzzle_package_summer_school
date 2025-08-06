import jax.numpy as jnp
import matplotlib.pyplot as plt

# ==================
colors = jnp.array([
    [0, 0, 0],   # Red
    [0, 1, 0],   # Green
    [0, 0, 1],   # Blue
    [1, 1, 0],   # Yellow
    [1, 0, 1],   # Magenta
])


def plt_imshow(puzzle, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    # Initialize blank RGB image
    H, W = puzzle.shape[1], puzzle.shape[2]
    rgb_image = jnp.zeros((H, W, 3))

    # Add each mask's color where it's True
    for i in range(puzzle.shape[0]):
        mask = puzzle[i]  # shape (H, W)
        for c in range(3):
            rgb_image = rgb_image.at[:, :, c].add(mask * colors[i, c])

    # Clip to [0, 1] in case of overlapping True values
    rgb_image = jnp.clip(rgb_image, 0, 1)

    # Plot
    ax.imshow(rgb_image)
    ax.set_axis_off()
