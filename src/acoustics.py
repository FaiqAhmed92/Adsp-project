import numpy as np
from scipy.signal import fftconvolve


class RoomAcoustics:
    def __init__(self, room_dims, source_positions, receiver_positions, abs_coeff, max_order, fs=44100):
        self.room_dims = room_dims
        self.source_positions = source_positions
        self.receiver_positions = receiver_positions
        self.abs_coeff = abs_coeff  # Absorption coefficients for low, mid, high frequencies
        self.max_order = max_order
        self.c = 343  # Speed of sound in m/s
        self.fs = fs  # Sampling frequency in Hz

    def image_source_method(self, enhanced=False):
        """Compute impulse responses for low, mid, and high frequencies separately."""
        impulse_responses = {"low": [], "mid": [], "high": []}

        for src_pos in self.source_positions:
            for rec_pos in self.receiver_positions:
                response_low = np.zeros(int(self.fs * 1.5))  # 1.5 sec duration
                response_mid = np.zeros(int(self.fs * 1.5))
                response_high = np.zeros(int(self.fs * 1.5))

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

                            if time_delay * self.fs < len(response_low):
                                idx = int(time_delay * self.fs)

                                # Compute reflection loss separately for each frequency band
                                loss_low = self.calculate_reflection_loss(order_x, order_y, order_z, "low", enhanced)
                                loss_mid = self.calculate_reflection_loss(order_x, order_y, order_z, "mid", enhanced)
                                loss_high = self.calculate_reflection_loss(order_x, order_y, order_z, "high", enhanced)

                                response_low[idx] += loss_low / dist
                                response_mid[idx] += loss_mid / dist
                                response_high[idx] += loss_high / dist

                impulse_responses["low"].append(response_low)
                impulse_responses["mid"].append(response_mid)
                impulse_responses["high"].append(response_high)

        return impulse_responses

    def calculate_reflection_loss(self, order_x, order_y, order_z, freq_band, enhanced):
        """Calculate reflection loss for a specific frequency band (low, mid, high)."""
        base_abs_coeff = self.abs_coeff[freq_band][0]  # Assuming same coeff for all surfaces

        if enhanced:
            base_abs_coeff += 0.05 * (abs(order_x) + abs(order_y) + abs(order_z))
            base_abs_coeff = min(base_abs_coeff, 0.9)  # Cap max absorption

        reflection_loss = (1 - base_abs_coeff) ** (abs(order_x) + abs(order_y) + abs(order_z))
        return reflection_loss
