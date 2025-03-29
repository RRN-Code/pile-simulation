import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(
    page_title="Pile Simulation",
    page_icon="assets/icon.png",
    layout="wide"
)

# ---------------------------------------------
# Class Definition: PreHomogenizationPileSimulation
# ---------------------------------------------
class PreHomogenizationPileSimulation:
    def __init__(self, pile_length, stacker_velocity, production_rate, total_weight, 
                 truck_payload, ore_grades):
        """
        Parameters:
            pile_length : float
                Total length of the pile (m) – this will be the x axis.
            stacker_velocity : float
                Horizontal velocity of the stacker (m/min).
            production_rate : float
                Production rate (tons/hour).
            total_weight : float
                Total weight to be piled (tons).
            truck_payload : float
                Weight per truck load (tons).
            ore_grades : dict
                Dictionary mapping source names to ore grade values.
        """
        self.pile_length = pile_length
        self.stacker_velocity = stacker_velocity  # m/min
        self.production_rate = production_rate    # tons/hour
        self.total_weight = total_weight
        self.truck_payload = truck_payload
        self.ore_grades = ore_grades
        
        self.material_density = 2.0  # tons/m³
        self.pile_width = 10.0       # Default width in meters
        
        # Discretize x-axis for the pile profile
        self.grid_resolution = 0.5
        self.x = np.arange(0, pile_length + self.grid_resolution, self.grid_resolution)
        self.height = np.zeros_like(self.x)
        
        # List to store each layer section
        self.layer_sections = []
        
        # Current stacker position, direction, and layer level
        self.stacker_position = 0.0
        self.forward = True
        self.current_layer = 0  # Layer index (increments when changing direction)
        
        # Store stacker positions and heights for reference
        self.stacker_positions = []
        self.stacker_heights = []
        
        # Track the current layer height at each position
        self.layer_heights = [0.0]  # Height at the start of each layer
    
    def add_material_section(self, material_tonnage, source):
        """
        Add a new layer section based on material tonnage and stacker velocity.
        Uses a back-and-forth movement, stacking layers on top of each other.
        """
        tons_per_minute = self.production_rate / 60.0
        deposition_time = material_tonnage / tons_per_minute
        section_length = self.stacker_velocity * deposition_time
        x_start = self.stacker_position
        
        if self.forward:
            x_end = x_start + section_length
            if x_end > self.pile_length:
                # Exceeds the pile length; handle partial deposit and direction flip
                excess = x_end - self.pile_length
                x_end = self.pile_length
                self.forward = False
                main_section_material = material_tonnage * (1 - excess/section_length)
                self.add_section_at_position(x_start, x_end, main_section_material, source)
                self.current_layer += 1
                previous_height = self.layer_heights[-1]
                total_volume = 0
                prev_layer_sections = [s for s in self.layer_sections if s['layer'] == self.current_layer - 1]
                for section in prev_layer_sections:
                    sec_length = section['x_end'] - section['x_start']
                    section_volume = sec_length * self.pile_width * section['thickness']
                    total_volume += section_volume
                if total_volume > 0:
                    avg_thickness = total_volume / (self.pile_length * self.pile_width)
                    new_height = previous_height + avg_thickness
                    self.layer_heights.append(new_height)
                else:
                    self.layer_heights.append(previous_height)
                remaining_material = material_tonnage - main_section_material
                self.stacker_position = self.pile_length
                self.add_material_section(remaining_material, source)
                return
            self.stacker_position = x_end
        else:
            x_end = x_start - section_length
            if x_end < 0:
                # Exceeds the left boundary
                excess = -x_end
                x_end = 0
                self.forward = True
                main_section_material = material_tonnage * (1 - excess/section_length)
                self.add_section_at_position(x_start, x_end, main_section_material, source)
                self.current_layer += 1
                previous_height = self.layer_heights[-1]
                total_volume = 0
                for section in [s for s in self.layer_sections if s['layer'] == self.current_layer - 1]:
                    sec_length = section['x_end'] - section['x_start']
                    section_volume = sec_length * self.pile_width * section['thickness']
                    total_volume += section_volume
                if total_volume > 0:
                    avg_thickness = total_volume / (self.pile_length * self.pile_width)
                    new_height = previous_height + avg_thickness
                    self.layer_heights.append(new_height)
                else:
                    self.layer_heights.append(previous_height)
                remaining_material = material_tonnage - main_section_material
                self.stacker_position = 0
                self.add_material_section(remaining_material, source)
                return
            self.stacker_position = x_end
        
        self.add_section_at_position(x_start, x_end, material_tonnage, source)
    
    def add_section_at_position(self, x_start, x_end, material_tonnage, source):
        """
        Add a section at the specified position with the given material.
        """
        x_start_actual = min(x_start, x_end)
        x_end_actual = max(x_start, x_end)
        section_length = x_end_actual - x_start_actual
        if section_length <= 0:
            return
        
        material_volume = material_tonnage / self.material_density
        section_thickness = material_volume / (section_length * self.pile_width)
        
        indices = np.where((self.x >= x_start_actual) & (self.x <= x_end_actual))[0]
        if len(indices) == 0:
            return
        
        current_height = self.layer_heights[self.current_layer]
        layer_info = {
            'x_start': x_start_actual,
            'x_end': x_end_actual,
            'y_bottom': current_height,
            'thickness': section_thickness,
            'source': source,
            'layer': self.current_layer
        }
        self.layer_sections.append(layer_info)
        self.height[indices] = current_height + section_thickness
        
        self.stacker_positions.append(self.stacker_position)
        self.stacker_heights.append(current_height + section_thickness)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def get_deposit_pattern_truck(ore_grades, deposit_ratio_choice):
    """
    Returns a deposit pattern list for the truck process.
    For two ore grades:
        "1:1" returns ['A', 'B']
        "2:1" returns ['A', 'A', 'B']
    For three ore grades:
        "1:1:1" returns [lowest, middle, highest]
        "2:1:1" returns [lowest, lowest, middle, highest]
    """
    keys = list(ore_grades.keys())
    if len(keys) == 2:
        if deposit_ratio_choice == "1:1":
            return ['A', 'B']
        elif deposit_ratio_choice == "2:1":
            return ['A', 'A', 'B']
        else:
            return ['A', 'B']
    elif len(keys) == 3:
        # Sort keys in increasing order of ore grade value
        sorted_keys = ['A', 'B', 'C']
        if deposit_ratio_choice == "1:1:1":
            return sorted_keys
        elif deposit_ratio_choice == "2:1:1":
            return ['A', 'A', 'B', 'C']
        else:
            return sorted_keys
    else:
        # Default fallback
        return keys

def truck_process(env, sim, deposit_pattern):
    """
    SimPy process to simulate trucks arriving with a given payload and deposit pattern.
    """
    truck_interval_minutes = (sim.truck_payload / sim.production_rate) * 60  
    truck_interval_seconds = truck_interval_minutes * 60
    truck_count = int(sim.total_weight / sim.truck_payload)
    
    pattern_length = len(deposit_pattern)
    deposit_index = 0
    
    for i in range(truck_count):
        source = deposit_pattern[deposit_index]
        deposit_index = (deposit_index + 1) % pattern_length
        sim.add_material_section(sim.truck_payload, source)
        yield env.timeout(truck_interval_seconds)

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
    # ax.set_title("Final Pre-Homogenization Pile Cross-Section")
    
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
        # sorted_keys = sorted(sim.ore_grades, key=lambda k: sim.ore_grades[k])
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

    # # Use np.isclose for robust floating point comparison
    # mask_A = np.isclose(deposit_values, default_grade_A)
    # deposit_values[mask_A] = deposit_values[mask_A] * np.random.uniform(0.9, 1.1, size=mask_A.sum())

    # mask_B = np.isclose(deposit_values, default_grade_B)
    # deposit_values[mask_B] = deposit_values[mask_B] * np.random.uniform(0.95, 1.05, size=mask_B.sum())

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
    # ax_mean.set_title("Mean Grade per Segment")
    ax_mean.set_xlabel("Pile Segment (m)")
    ax_mean.set_ylabel("Mean Grade (%)")
    ax_mean.set_ylim(bottom=max(0, min(column_means) * 0.67), top=max(column_means) * 1.5)
    ax_mean.grid(True, linestyle='--', alpha=0.3)

    # --- Plot 2: Standard Deviation ---
    fig_std, ax_std = plt.subplots(figsize=(10, 2))
    ax_std.plot(segment_indices, column_stds, color='red', marker='x', linestyle='--')
    ax_std.set_xticks(np.arange(0, len(segment_indices), 1))
    ax_std.set_xticklabels(np.arange(10, 10 * (len(segment_indices) + 1), 10), fontsize=10)
    # ax_std.set_title("Standard Deviation per Segment")
    ax_std.set_xlabel("Pile Segment (m)")
    ax_std.set_ylabel("Standard Deviation (%)")
    ax_std.set_ylim(bottom=max(0, min(column_stds) * 0.67), top=max(column_stds) * 1.5)
    ax_std.grid(True, linestyle='--', alpha=0.3)

    return fig_mean, fig_std, deposit_values, grade_matrix, column_means


# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------
def main():
    st.title("Pre-Homo Pile Simulation")
    st.sidebar.title("Simulation Parameters")

    # --- Sidebar inputs ---
    st.sidebar.markdown("### Basic Parameters")
    pile_length = st.sidebar.slider("Pile Length (m)", min_value=100.0, max_value=300.0, value=100.0, step=10.0)
    stacker_velocity = st.sidebar.slider("Stacker Velocity (m/min)", min_value=1.0, max_value=20.0, value=10.0, step=1.0)
    production_rate = st.sidebar.slider("Production Rate (tons/hour)", min_value=300.0, max_value=900.0, value=600.0, step=100.0)
    total_weight = st.sidebar.slider("Total Material (tons)", min_value=900.0, max_value=9000.0, value=900.0, step=100.0)
    truck_payload = st.sidebar.slider("Truck Payload (tons)", min_value=20.0, max_value=60.0, value=30.0, step=10.0)

    # --- Ore Source Grades and Variabilities ---
    st.sidebar.markdown("### Ore Source Zones")
    grade_option = st.sidebar.radio("Number of ore sources feeding the pile", ["2 Sources", "3 Sources"])

    if grade_option == "2 Sources":
        grade_a = st.sidebar.slider("Grade A (%)", min_value=1.0, max_value=50.0, value=30.0, step=1.0)
        var_a = st.sidebar.slider("Absolute Deviation A (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_b = st.sidebar.slider("Grade B (%)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        var_b = st.sidebar.slider("Absolute Deviation B (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        ore_grades = {'A': grade_a, 'B': grade_b}
        variabilities = {'A': var_a, 'B': var_b}
        deposit_ratio_choice = st.sidebar.selectbox("Deposit Ratio", ["1:1", "2:1"])

    else:
        grade_a = st.sidebar.slider("Grade A (%)", min_value=1.0, max_value=50.0, value=30.0, step=1.0)
        var_a = st.sidebar.slider("Absolute Deviation A (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_b = st.sidebar.slider("Grade B (%)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        var_b = st.sidebar.slider("Absolute Deviation B (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_c = st.sidebar.slider("Grade C (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        var_c = st.sidebar.slider("Absolute Deviation C (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        ore_grades = {'A': grade_a, 'B': grade_b, 'C': grade_c}
        variabilities = {'A': var_a, 'B': var_b, 'C': var_c}
        deposit_ratio_choice = st.sidebar.selectbox("Deposit Ratio", ["1:1:1", "2:1:1"])

    # --- Run simulation ---
    deposit_pattern_truck = get_deposit_pattern_truck(ore_grades, deposit_ratio_choice)
    sim = PreHomogenizationPileSimulation(
        pile_length, stacker_velocity, production_rate,
        total_weight, truck_payload, ore_grades
    )

    for source in deposit_pattern_truck * int(sim.total_weight // sim.truck_payload):
        sim.add_material_section(sim.truck_payload, source)

    # --- Display plots and metrics ---
    fig_mean, fig_std, deposit_values, grade_matrix, column_means = plot_grade_matrix(sim, deposit_ratio_choice, variabilities)



    col_plot, col_metrics = st.columns(2)

    with col_plot:
        st.subheader("Stacking: Longitudinal Section")
        st.pyplot(plot_final_pile(sim, deposit_ratio_choice, variabilities))


    with col_metrics:
        st.subheader("Reclaiming: Segment Statistics")
        st.pyplot(fig_mean)
        st.pyplot(fig_std)

        mean_val = np.mean(column_means)
        std_val = np.std(column_means)

        st.subheader("Reclaiming: Overall Performance")
        space1, col1, space2, col2, space3 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown(" ")
            st.metric("Mean Grade (%)", f"{mean_val:.2f}")
        with col2:
            st.markdown(" ")
            st.metric("Std Dev (%)", f"{std_val:.2f}")

    st.markdown("---")

    image1, image2 = st.columns(2)
    with image1:
        st.subheader("Chevron Stacking/Reclaiming Layout")
        st.image("assets/chevron.png", use_container_width=True)
        st.markdown(
            "Chevron stacking forms a central ridge. The stacker reverses direction to build layers upward, "
            "achieving vertical buildup with alternating passes."
        )

    with image2:
        st.subheader("Stacker - Sequential Layer Formation")
        st.image("assets/pre_homo_pile.gif", use_container_width=True)


    # st.markdown("---")

    # st.markdown("Grade Matrix (First 10 layers at the bottom)")
    # st.dataframe(grade_matrix[:10])
    # st.dataframe(grade_matrix[-10:])

if __name__ == "__main__":
    main()
