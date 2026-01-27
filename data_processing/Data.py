class Data():
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.label = label
        self.sampling_frequency = sampling_frequency
        self.f_mains = f_mains
        self.validate()
    
    def validate(self):
        # validates signal data integrity
        if len(self.current_segment) != len(self.voltage_segment):
            raise ValueError(f"Current and voltage must have same length: {len(self.current_segment)} vs {len(self.voltage_segment)}")
        if len(self.current_segment) == 0:
            raise ValueError("Signals cannot be empty")
        if self.sampling_frequency <= 0:
            raise ValueError(f"Invalid sampling frequency: {self.sampling_frequency}")
        if self.f_mains not in [50, 60]:
            raise ValueError(f"Mains frequency must be 50 or 60 Hz, got: {self.f_mains}")
        return True
    
    def get_points_per_cycle(self):
        # returns number of sample points per mains cycle
        return int(self.sampling_frequency / self.f_mains)
    
    def get_time_step(self):
        # returns time interval between samples (in seconds)
        return 1 / self.sampling_frequency
    
    def __repr__(self):
        # string representation for debugging
        duration = len(self.current_segment) * self.get_time_step()
        return (f"Data(label='{self.label}', "
                f"samples={len(self.current_segment)}, "
                f"fs={self.sampling_frequency}Hz, "
                f"f_mains={self.f_mains}Hz, "
                f"duration={duration:.3f}s)")