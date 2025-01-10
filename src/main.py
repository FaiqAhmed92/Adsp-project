# src/main.py
import json
import os
from pathlib import Path
from acoustics import RoomAcoustics
import matplotlib.pyplot as plt
def load_room_specs(file_path):
    """Load room specifications from a JSON file test."""
    with open(file_path, 'r') as file:
        return json.load(file)

def simulate_acoustics(acoustics_simulator):
    """Simulate both original and extended impulse responses."""
    original_responses = acoustics_simulator.image_source_method()
    extended_responses = acoustics_simulator.image_source_method(enhanced=True)
    return original_responses, extended_responses

def plot_responses(original_responses, extended_responses, title):
    """Plot both original and extended impulse responses for comparison."""
    plt.figure(figsize=(10, 5))
    plt.plot(original_responses[0], label='Original Response', linestyle='--')
    plt.plot(extended_responses[0], label='Extended Response', linestyle='-')
    plt.title(f"Comparison of Original and Extended Impulse Responses - {title}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Directory containing JSON configurations
    config_path = 'C:\\Users\\LENOVO\\Desktop\\Website\\adsp-master\\data'
    room_configs = ["small_room.json", "large_room.json", "complex_room.json"]
    
    for config_file in room_configs:
        full_path = os.path.join(config_path, config_file)
        room_specs = load_room_specs(full_path)
        
        # Create a RoomAcoustics object with loaded specifications
        acoustics_simulator = RoomAcoustics(
            room_dims=room_specs['room_dims'],
            source_positions=room_specs['source_positions'],
            receiver_positions=room_specs['receiver_positions'],
            abs_coeff=room_specs['abs_coeff'],
            max_order=room_specs['max_order']
        )
        
        # Simulate acoustic responses
        original_responses, extended_responses = simulate_acoustics(acoustics_simulator)
        
        # Plot responses
        plot_responses(original_responses, extended_responses, Path(config_file).stem)

if __name__ == "__main__":
    main()