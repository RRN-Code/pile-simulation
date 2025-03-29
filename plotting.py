import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ----------------------------------------------------
# Plotting Functions
# ----------------------------------------------------
def plot_final_pile(sim, deposit_ratio_choice, variabilities):
    """
    Plot the final cross-section of the pile using Matplotlib,
    returning the figure object for Streamlit.
    """
    # Decide on color map for 2 or 3 ore grades
    if len(sim.ore_grades) == 2:
        color_map = {'A': 'blue', 'B': 'red'}
    else:
        color_map = {'A': 'blue', 'B': 'red', 'C': 'green'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute max height for plot limits
    if sim.layer_sections:
        max_sections = [section['y_bottom'] + section['thickness'] for section in sim.layer_sections]
        max_height = max(max_sections) if max_sections else 1
    else:
        max_height = 1
    
    ax.set_xlim(0, sim.pile_length)
    ax.set_xticks(np.arange(0, sim.pile_length + 1, 10))
    ax.set_ylim(0, max_height * 1.5)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Height (m)")
    
    # Draw each layer section
    for section in sim.layer_sections:
        rect = patches.Rectangle(
            (section['x_start'], section['y_bottom']),
            section['x_end'] - section['x_start'],
            section['thickness'],
            facecolor=color_map.get(section['source'], 'gray'),
            edgecolor='black',
            linewidth=0.2,
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Overplot the final pile profile
    ax.plot(sim.x, sim.height, color='black', linewidth=1.5)
    
    # Vertical grid lines
    for x_line in range(0, int(sim.pile_length)+1, 10):
        ax.axvline(x=x_line, color='gray', linestyle='--', alpha=0.5)

    # Legend for sources with variability
    handles = [
        patches.Patch(
            color=color_map[s],
            label=f"Source {s} (Grade: {sim.ore_grades[s]}%, Var: {variabilities.get(s, 0)}%)"
        )
        for s in sim.ore_grades
    ]
    ax.legend(handles=handles, loc='upper right')
   
    # Parameter text
    param_text = (
        f"Pile Length: {sim.pile_length} m\n"
        f"Stacker Velocity: {sim.stacker_velocity} m/min\n"
        f"Production Rate: {sim.production_rate} tons/hr\n"
        f"Truck Payload: {sim.truck_payload} tons\n"
        f"Total Material: {sim.total_weight} tons\n"
        f"Deposit Proportion: {deposit_ratio_choice}"
    )
    ax.text(0.02, 0.97, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig

def plot_grade_matrix(sim, deposit_ratio_choice, variabilities):
    """
    Returns two separate plots: one for mean grade, one for standard deviation per segment.
    """
    pile_length = sim.pile_length
    segment_length = 10.0
    num_segments = int(pile_length / segment_length)
    num_layers = len(sim.layer_heights)
    total_cells = num_layers * num_segments

    keys = list(sim.ore_grades.keys())

    # Build numeric pattern based on grade values
    if len(keys) == 2:
        if deposit_ratio_choice == "1:1":
            pattern = [sim.ore_grades['A']] * 3 + [sim.ore_grades['B']] * 3
        elif deposit_ratio_choice == "2:1":
            pattern = [sim.ore_grades['A']] * 6 + [sim.ore_grades['B']] * 3
        else:
            pattern = [sim.ore_grades['A']] * 3 + [sim.ore_grades['B']] * 3
    elif len(keys) == 3:
        if deposit_ratio_choice == "1:1:1":
            pattern = (
                [sim.ore_grades['A']] * 3 +
                [sim.ore_grades['B']] * 3 +
                [sim.ore_grades['C']] * 3
            )
        elif deposit_ratio_choice == "2:1:1":
            pattern = (
                [sim.ore_grades['A']] * 6 +
                [sim.ore_grades['B']] * 3 +
                [sim.ore_grades['C']] * 3
            )
        else:
            pattern = (
                [sim.ore_grades['A']] * 6 +
                [sim.ore_grades['B']] * 3 +
                [sim.ore_grades['C']] * 3
            )
    else:
        pattern = []
        for s in list(sim.ore_grades.keys()):
            pattern.extend([sim.ore_grades[s]] * 3)

    repeats = (total_cells + len(pattern) - 1) // len(pattern)
    deposit_values = (pattern * repeats)[:total_cells]
    deposit_values = np.array(deposit_values)[::-1]

    # Retrieve the default grade values from the simulation object
    default_grade_A = sim.ore_grades['A']
    default_grade_B = sim.ore_grades['B']

    # Apply variability if defined
    for source, default_grade in sim.ore_grades.items():
        variability = variabilities.get(source, 0.0)  # in percent

        if variability > 0:
            lower = 1.0 - variability / 100.0
            upper = 1.0 + variability / 100.0
            mask = np.isclose(deposit_values, default_grade)
            deposit_values[mask] = deposit_values[mask] * np.random.uniform(lower, upper, size=mask.sum())

    grade_matrix = np.full((num_layers, num_segments), np.nan)
    coords = []
    for r in range(num_layers):
        if r % 2 == 0:
            for c in range(num_segments):
                coords.append((r, c))
        else:
            for c in range(num_segments - 1, -1, -1):
                coords.append((r, c))

    for i, (r, c) in enumerate(coords):
        grade_matrix[r, c] = deposit_values[i]

    column_means = np.nanmean(grade_matrix, axis=0)
    column_stds = np.nanstd(grade_matrix, axis=0)
    segment_indices = np.arange(num_segments)

    # --- Plot 1: Mean Grade ---
    fig_mean, ax_mean = plt.subplots(figsize=(9.5, 2))
    ax_mean.plot(segment_indices, column_means, color='blue', marker='o')
    ax_mean.set_xticks(np.arange(0, len(segment_indices), 1))
    ax_mean.set_xticklabels(np.arange(10, 10 * (len(segment_indices) + 1), 10), fontsize=10)
    ax_mean.set_xlabel("Pile Segment (m)")
    ax_mean.set_ylabel("Mean Grade (%)")
    ax_mean.set_ylim(bottom=max(0, min(column_means) * 0.9), top=max(column_means) * 1.1)
    ax_mean.grid(True, linestyle='--', alpha=0.3)

    # --- Plot 2: Standard Deviation ---
    fig_std, ax_std = plt.subplots(figsize=(10, 2))
    ax_std.plot(segment_indices, column_stds, color='red', marker='x', linestyle='--')
    ax_std.set_xticks(np.arange(0, len(segment_indices), 1))
    ax_std.set_xticklabels(np.arange(10, 10 * (len(segment_indices) + 1), 10), fontsize=10)
    ax_std.set_xlabel("Pile Segment (m)")
    ax_std.set_ylabel("Standard Deviation (%)")
    ax_std.set_ylim(bottom=max(0, min(column_stds) * 0.9), top=max(column_stds) * 1.1)
    ax_std.grid(True, linestyle='--', alpha=0.3)

    return fig_mean, fig_std, deposit_values, grade_matrix, column_means

