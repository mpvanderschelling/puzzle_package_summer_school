import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def filter_valid_path(full_path):
    # full_path shape: (max_steps+1, 2)
    # Valid steps have non-negative coordinates, invalid ones are (-1, -1)
    valid_mask = (full_path[:, 0] >= 0) & (full_path[:, 1] >= 0)
    return full_path[valid_mask]


def plot_visited_grid(full_path, chunks, grid_size=10, ax=None):
    start_matrix = jnp.zeros((grid_size, grid_size))

    idx_start = 0
    for i, idx_end in enumerate(chunks, 1):
        if idx_end == idx_start:
            break
        sl = full_path[idx_start:idx_end]
        idx_start = idx_end
        for x, y in sl:
            start_matrix = start_matrix.at[x, y].set(i)

    # Custom colormap: black for 0, then tab20 colors for others
    cmap = plt.cm.tab20
    new_colors = list(cmap.colors)  # Convert to list
    black = (0.0, 0.0, 0.0)
    colors = [black] + new_colors

    custom_cmap = mcolors.ListedColormap(colors)

    ax.imshow(start_matrix, cmap=custom_cmap, origin='lower')
    ax.set_axis_off()

    # path = filter_valid_path(full_path)
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(5, 5))
    # visited_grid = np.zeros((grid_size, grid_size), dtype=bool)
    # for x, y in path:
    #     visited_grid[int(x), int(y)] = True

    # ax.imshow(visited_grid.T, cmap='gray', origin='lower')
    # ax.set_axis_off()
