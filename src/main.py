import json
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from acoustics import RoomAcoustics  # Your existing class

# ===================== RT60 CALCULATION =====================
def compute_room_volume(room_dims):
    """Calculate room volume (length × width × height)."""
    length, width, height = room_dims
    return length * width * height

def compute_surface_areas(room_dims):
    """Calculate surface areas of the 6 sides of a rectangular room."""
    length, width, height = room_dims
    return [length * width, length * width,  # Floor & Ceiling
            length * height, length * height,  # Front & Back Walls
            width * height, width * height]  # Left & Right Walls

def calculate_rt60(room_dims, abs_coeff):
    """
    Compute RT60 for each frequency band using Sabine’s formula.
    abs_coeff should be a dictionary with absorption coefficients for 'low', 'mid', and 'high' frequencies.
    """
    volume = compute_room_volume(room_dims)
    surface_areas = compute_surface_areas(room_dims)

    rt60_results = {}
    for band, coeffs in abs_coeff.items():
        total_absorption = sum(area * alpha for area, alpha in zip(surface_areas, coeffs))
        rt60_results[band] = 0.161 * (volume / total_absorption) if total_absorption > 0 else 0
    return rt60_results

# ===================== ROOM & ACOUSTICS SIMULATION =====================
def load_room_specs(file_path):
    """Load room specifications from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def simulate_acoustics(acoustics_simulator):
    """Simulate impulse responses for different frequency bands."""
    return acoustics_simulator.image_source_method(enhanced=True)

def plot_3d_room(room_dims, source_positions, receiver_positions, title):
    """Plot a 3D representation of the room with sources and receivers."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x_dim, y_dim, z_dim = room_dims
    corners = np.array([[0, 0, 0], [x_dim, 0, 0], [x_dim, y_dim, 0], [0, y_dim, 0],
                        [0, 0, z_dim], [x_dim, 0, z_dim], [x_dim, y_dim, z_dim], [0, y_dim, z_dim]])

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for (start, end) in edges:
        ax.plot([corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]], color='black', linewidth=1)

    for src in source_positions:
        ax.scatter(src[0], src[1], src[2], color='red', marker='o', s=50, label='Source')

    for rec in receiver_positions:
        ax.scatter(rec[0], rec[1], rec[2], color='blue', marker='x', s=50, label='Receiver')

    ax.set_title(f"3D Room Layout - {title}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-1, x_dim + 1])
    ax.set_ylim([-1, y_dim + 1])
    ax.set_zlim([-1, z_dim + 1])
    ax.legend()
    plt.show()

def plot_responses_and_rt60(responses, rt60_values, title):
    """Plot impulse responses (2D), RT60 values, and Decay Curves."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    for freq_band, response in responses.items():
        ax1.plot(response[0], label=f'{freq_band.capitalize()} Frequency')

    ax1.set_xlabel("Time (samples)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Impulse Response - {title}")
    ax1.legend()
    ax1.grid(True)
    plt.show()

    # Plot RT60 values as a bar chart
    fig, ax2 = plt.subplots(figsize=(6, 4))
    bands = list(rt60_values.keys())
    values = list(rt60_values.values())
    ax2.bar(bands, values, color=['red', 'green', 'blue'])
    ax2.set_title(f"RT60 (Reverberation Time) - {title}")
    ax2.set_ylabel("RT60 (seconds)")
    plt.show()

    # Plot Decay Curves (log-scale)
    fig, ax3 = plt.subplots(figsize=(10, 5))
    for freq_band, response in responses.items():
        energy = np.cumsum(response[0]**2)[::-1]  # Reverse cumulative sum for decay
        energy_db = 10 * np.log10(energy / np.max(energy))
        ax3.plot(energy_db, label=f'{freq_band.capitalize()} Frequency')

    ax3.set_xlabel("Time (samples)")
    ax3.set_ylabel("Energy Decay (dB)")
    ax3.set_title(f"Energy Decay Curve (Log-Scale) - {title}")
    ax3.legend()
    ax3.grid(True)
    plt.show()

# ===================== MAIN FUNCTION =====================
def main():
    config_path = 'C:\\Users\\LENOVO\\Desktop\\Website\\adsp-master\\data'
    room_configs = [
        "small_room.json", "large_room.json", "complex_room.json",
        "furnished_room.json", "empty_room.json", "treated_big_room.json", "untreated_big_room.json"
    ]

    for config_file in room_configs:
        full_path = os.path.join(config_path, config_file)
        room_specs = load_room_specs(full_path)

        # Initialize RoomAcoustics
        acoustics_simulator = RoomAcoustics(
            room_dims=room_specs['room_dims'],
            source_positions=room_specs['source_positions'],
            receiver_positions=room_specs['receiver_positions'],
            abs_coeff=room_specs['abs_coeff'],
            max_order=room_specs['max_order']
        )

        # Compute RT60 values
        rt60_values = calculate_rt60(
            room_dims=room_specs['room_dims'],
            abs_coeff=room_specs['abs_coeff']
        )

        # Print RT60 results
        print(f"\n=== {config_file} ===")
        for band, rt60 in rt60_values.items():
            print(f"RT60 ({band}): {rt60:.3f} seconds")

        # Plot the 3D room layout
        plot_3d_room(
            room_dims=room_specs['room_dims'],
            source_positions=room_specs['source_positions'],
            receiver_positions=room_specs['receiver_positions'],
            title=Path(config_file).stem
        )

        # Simulate and plot impulse responses, RT60, and Decay Curves
        responses = simulate_acoustics(acoustics_simulator)
        plot_responses_and_rt60(responses, rt60_values, Path(config_file).stem)

if __name__ == "__main__":
    main()
