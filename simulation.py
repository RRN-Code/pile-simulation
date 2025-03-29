import numpy as np

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
