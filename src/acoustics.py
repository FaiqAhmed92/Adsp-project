import numpy as np
from scipy.signal import fftconvolve

class RoomAcoustics:
    def __init__(self, room_dims, source_positions, receiver_positions, abs_coeff, max_order, fs=44100):
        self.room_dims = room_dims
        self.source_positions = source_positions
        self.receiver_positions = receiver_positions
        self.abs_coeff = abs_coeff
        self.max_order = max_order
        self.c = 343  # Speed of sound in m/s
        self.fs = fs  # Sampling frequency in Hz

    def image_source_method(self, enhanced=False):
        impulse_responses = []
        for src_pos in self.source_positions:
            for rec_pos in self.receiver_positions:
                impulse_response = np.zeros(int(self.fs * 1.5))  # 1.5 seconds duration
                for order_x in range(-self.max_order, self.max_order + 1):
                    for order_y in range(-self.max_order, self.max_order + 1):
                        for order_z in range(-self.max_order, self.max_order + 1):
                            mirrored_src = [
                                src_pos[0] + 2 * order_x * self.room_dims[0],
                                src_pos[1] + 2 * order_y * self.room_dims[1],
                                src_pos[2] + 2 * order_z * self.room_dims[2],
                            ]
                            dist = np.linalg.norm(np.array(mirrored_src) - np.array(rec_pos))
                            time_delay = dist / self.c
                            if time_delay * self.fs < len(impulse_response):
                                idx = int(time_delay * self.fs)
                                reflection_loss = self.calculate_reflection_loss(order_x, order_y, order_z, enhanced)
                                impulse_response[idx] += reflection_loss / dist
                impulse_responses.append(impulse_response)
        return impulse_responses

    def calculate_reflection_loss(self, order_x, order_y, order_z, enhanced):
        if enhanced:
            # Higher order reflections suffer more loss
            abs_coeff_base = self.abs_coeff["mid"][0] + 0.05 * (abs(order_x) + abs(order_y) + abs(order_z))
            abs_coeff_base = min(abs_coeff_base, 0.9)  # Cap the absorption to not exceed 90%
        else:
            abs_coeff_base = self.abs_coeff["mid"][0]
        
        reflection_loss = (1 - abs_coeff_base) ** (abs(order_x) + abs(order_y) + abs(order_z))
        return reflection_loss

