import jax.numpy as jnp
import matplotlib.pyplot as plt

# ==================

COLORMAPS = [
    'grey',
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
]

colors = jnp.array([
    [0, 0, 0],   # Red
    [0, 1, 0],   # Green
    [0, 0, 1],   # Blue
    [1, 1, 0],   # Yellow
    [1, 0, 1],   # Magenta
    [0, 1, 1],   # Cyan
    [1, 0.5, 0],  # Orange
    [0.5, 0, 1],  # Purple
    [0, 0.5, 0.5],  # Teal
    [0.5, 0.5, 0],  # Olive
    [0.5, 0, 0],  # Maroon
    [0, 0.5, 0],  # Dark Green
    [0, 0, 0.5],  # Navy
    [1, 0.75, 0.8],  # Pink
    [0.6, 0.4, 0.2],  # Brown
    [0.3, 0.3, 0.3],  # Dark Grey
])

# colors = jnp.array([
#     [0.00, 0.60, 1.00],  # Bright Blue
#     [1.00, 0.60, 0.00],  # Bright Orange
#     [0.00, 0.80, 0.60],  # Bright Teal
#     [1.00, 1.00, 0.20],  # Bright Yellow
#     [1.00, 0.30, 0.30],  # Bright Red (still distinguishable)
# ])


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
