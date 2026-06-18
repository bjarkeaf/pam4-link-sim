#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, bessel

#%% Utility functions for LFSR and PRBS generation

def lfsr_sequence(register_size, taps, num_bits, seed = None):
    """Generates a sequence of bits using a Linear Feedback Shift Register (LFSR).

    Args:
        register_size (int): The size of the LFSR register.
        taps (list of int): The positions of the taps in the LFSR (0-indexed from the most significant bit).
        seed (int): The initial state of the LFSR.
        num_bits (int): The number of bits to generate.

    Returns:
        list of int: A list containing the generated bits.
    """
    state = seed if seed is not None else 1  # Default seed value if not provided

    bits = []
    for _ in range(num_bits):
        # Calculate the feedback/output bit
        feedback_bit = 0
        for tap in taps:
            lsb_tap = register_size - 1 - tap  # Convert tap position to match the bit indexing
            feedback_bit ^= (state >> lsb_tap) & 1  # XOR the tapped bits

        # Shift the register to the right and insert the feedback bit at the leftmost position (MSB)
        state = (state >> 1) | (feedback_bit << (register_size - 1))
        bits.append(feedback_bit) # Append the output bit (feedback bit) to the sequence

    return bits

def prbs_sequence(register_size, seed, num_bits=None, taps=None):
    """Generates a Pseudo-Random Binary Sequence (PRBS) using an LFSR.

    Args:
        register_size (int): The size of the LFSR register.
        seed (int): The initial state of the LFSR.
        num_bits (int): The number of bits to generate.
        taps (list of int, optional): Custom tap positions. If not provided, uses standard PRBS taps
                                      for supported register sizes (7, 9, 11, 13, 15, 20, 23, 31).
                                      If register size not in supported list and taps not provided,
                                      defaults to empty list (no feedback).
    Returns:
        list of int: A list containing the generated PRBS bits.
    """
    # Tap positions are 0-indexed from the MSB, so tap 0 corresponds to tapping between the two leftmost bits.
    # Tap 0 corresponds to the x term in the polynomial, and the highest tap corresponds to the x^n term where n is the register size.
    standard_taps = {
        7:  [5, 6],            # x^7 + x^6 + 1
        9:  [4, 8],            # x^9 + x^5 + 1
        11: [8, 10],           # x^11 + x^9 + 1
        13: [0, 1, 11, 12],    # x^13 + x^12 + x^2 + x + 1
        15: [13, 14],          # x^15 + x^14 + 1
        20: [2, 19],           # x^20 + x^3 + 1
        23: [17, 22],          # x^23 + x^18 + 1
        31: [27, 30],          # x^31 + x^28 + 1
    }

    # Use provided num_bits or default to a full cycle length for the given register size (2^n - 1)
    if num_bits is None:
        num_bits = (1 << register_size) - 1  # 2^n - 1

    # Use provided taps or default to standard PRBS if available
    if taps is None:
        taps = standard_taps.get(register_size, [])
    
    return lfsr_sequence(register_size=register_size, taps=taps, seed=seed, num_bits=num_bits)

def gray_code_pairs(bit_sequence):
    """Pairs consecutive bits and Gray-codes them into integers (0-3).
    Handles odd-length input naturally by concatenating two repetitions,
    per IEEE 802.3 120.5.11.2.1 (the pairing phase shift across repetitions
    falls out automatically from pairing the doubled sequence)."""
    sequence_length = len(bit_sequence)
    if sequence_length % 2 != 0:
        bit_sequence = bit_sequence + bit_sequence[:1] # Duplicate first bit to make length even
    gray_to_int = [0, 1, 3, 2]  # {0,0}->0 {0,1}->1 {1,0}->3 {1,1}->2
    return [gray_to_int[(bit_sequence[i] << 1) | bit_sequence[i+1]] for i in range(0, sequence_length, 2)]

# def precode(data_sequence, block_size=46):
#     # Not needed for test patterns, only for training pattern, but left for future use.
#     """Precodes a sequence of integer symbols (0-3) per IEEE 802.3 94.2.2.6, by subtracting 
#     the previous precoded symbol from the current data symbol modulo 4. First symbol of each
#     block of size block_size is passed through unchanged."""
#     if not data_sequence:
#         return []
#     precoded_sequence = []
#     for i in range(len(data_sequence)):
#         if i % block_size == 0:
#             precoded_bit = data_sequence[i]  # termination symbol, passes through
#         else:
#             precoded_bit = (data_sequence[i] - precoded_sequence[i - 1]) % 4
#         precoded_sequence.append(precoded_bit)
#     return precoded_sequence

# def int_to_pam4(integer_sequence):
#     # Unnecessary since we can directly map integers to power levels, but left for future use.
#     """Converts a sequence of integers (0, 1, 2, 3) to their corresponding PAM4 symbols (-1, -1/3, 1/3, 1)."""
#     symbol_levels = [-1, -1/3, 1/3, 1]
#     return [symbol_levels[i] for i in integer_sequence]

def symbol_to_power(symbol_sequence, power_levels=[0.333, 0.667, 1.0, 1.333]):
    """Converts an integer symbol to its corresponding power level in mW"""
    return [power_levels[symbol] for symbol in symbol_sequence]

def power_level_statistics(power_levels):
    """Reports derived TX metrics from the four PAM4 power levels [P0, P1, P2, P3].

    Returns a dict with:
        OMAouter  (mW)  -- outer modulation amplitude, P3 - P0
        ER_dB           -- extinction ratio in dB, 10*log10(P3/P0)
        RLM             -- level separation mismatch ratio (1.0 = ideal, min 0.92 per spec)
        ES1             -- inner level asymmetry, lower eye
        ES2             -- inner level asymmetry, upper eye
        avg_power (mW)  -- average optical power assuming equal symbol probabilities
    """
    P0, P1, P2, P3 = power_levels
    gaps = [P1 - P0, P2 - P1, P3 - P2]
    OMAouter = P3 - P0
    P_avg = np.mean(power_levels)
    return {
        "OMAouter_mW": OMAouter,
        "ER_dB":       10 * np.log10(P3 / P0),
        "RLM":         3 * min(gaps) / OMAouter,
        "ES1":         (P1 - P_avg) / (P0 - P_avg),
        "ES2":         (P2 - P_avg) / (P3 - P_avg),
        "avg_power_mW": P_avg,
    }

def upsample(symbol_sequence, samples_per_ui):
    """Repeats each symbol sample_per_ui times to go from 1 sample/symbol to N samples/UI."""
    return np.repeat(symbol_sequence, samples_per_ui).astype(float)

def fiber_loss_db(distance_m, attenuation_db_per_km=0.5, connection_loss_db=2.75):
    """Returns total fiber channel insertion loss in dB for a given distance, using a flat attenuation model."""
    return attenuation_db_per_km * distance_m / 1000 + connection_loss_db

def fiber_dispersion_ps_per_nm(distance_m, wavelength_nm=1310.0, lambda0_nm=1300.0, S0_ps_per_nm2_per_km=0.093):
    """Returns total chromatic dispersion D*L in ps/nm for a SMF link (IEC 60793-2-50 formula).

    Args:
        distance_m (float): Fiber length in meters.
        wavelength_nm (float): Laser wavelength in nm (default 1310).
        lambda0_nm (float): Zero-dispersion wavelength in nm (1300-1324 per Table 121-14; use 1300 for worst case).
        S0_ps_per_nm2_per_km (float): Dispersion slope in ps/nm²/km (max 0.093 per Table 121-14).

    Returns:
        float: Total dispersion D*L in ps/nm. Bounded by -0.93 to +0.8 ps/nm for 200GBASE-DR4 (Table 121-13).
    """
    D = (S0_ps_per_nm2_per_km / 4) * (wavelength_nm - lambda0_nm**4 / wavelength_nm**3)
    return D * distance_m / 1000

def channel_filter(waveform, samples_per_ui, ui, loss_db=0.0, dispersion_ps_per_nm=0.0, wavelength_nm=1310.0):
    """Applies fiber channel: flat attenuation and chromatic dispersion.

    Args:
        waveform (np.ndarray): TX waveform at N samples/UI.
        samples_per_ui (int): Oversampling factor N.
        ui (float): Symbol period in seconds (= 1 / symbol_rate).
        loss_db (float): Total channel insertion loss in dB (0-3 dB for 200GBASE-DR4).
        dispersion_ps_per_nm (float): Total chromatic dispersion D*L in ps/nm (-0.93 to +0.8 for 200GBASE-DR4).
        wavelength_nm (float): Laser center wavelength in nm (default 1310).

    Returns:
        np.ndarray: Waveform after channel, same shape as input.
    """
    # Attenuation: scale all power levels equally, ER unchanged
    out = waveform * 10 ** (-loss_db / 10)

    # Chromatic dispersion: quadratic phase in frequency domain
    # H(f) = exp(+j * pi * D_total * lambda^2 / c * f^2), Agrawal sign convention
    if dispersion_ps_per_nm != 0.0:
        n = len(out)
        dt = ui / samples_per_ui
        f = np.fft.fftfreq(n, d=dt)                          # baseband frequencies in Hz
        D_total = dispersion_ps_per_nm * 1e-12 / 1e-9        # convert ps/nm to s/m
        lam = wavelength_nm * 1e-9                           # m
        c = 3e8                                               # m/s
        H = np.exp(1j * np.pi * D_total * lam**2 / c * f**2)
        out = np.fft.ifft(np.fft.fft(out) * H).real

    return out

def tx_filter(waveform, samples_per_ui, transition_time, ui):
    """Applies a single-pole low-pass filter modelling finite TX bandwidth.

    Args:
        waveform (np.ndarray): Upsampled waveform at N samples/UI.
        samples_per_ui (int): Oversampling factor N (used to compute dt).
        transition_time (float): TX 20%-80% transition time in seconds (max 34e-12 for 26.5625 GBd).
        ui (float): Symbol period in seconds (= 1 / symbol_rate).

    Returns:
        np.ndarray: Filtered waveform, same shape as input.
    """
    tau = transition_time / np.log(4)   # t_rise = tau * ln(4)  →  tau = t_rise / ln(4)
    dt = ui / samples_per_ui
    alpha = dt / (tau + dt)             # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    b = [alpha]
    a = [1, -(1 - alpha)]               # lfilter subtracts a[1]*y[n-1], so a[1] is negated
    return lfilter(b, a, waveform)

def rx_filter(waveform, samples_per_ui, symbol_rate, order=4, f3db_hz=None):
    """Applies a Bessel-Thomson low-pass filter modeling the O/E converter and oscilloscope.

    Args:
        waveform (np.ndarray): Input waveform at N samples/UI.
        samples_per_ui (int): Oversampling factor N.
        symbol_rate (float): Symbol rate in Hz (e.g. 26.5625e9).
        order (int): Filter order (4 per IEEE 802.3-2022 121.8.5.1).
        f3db_hz (float): 3 dB bandwidth in Hz. Defaults to 0.5 * symbol_rate (~13.28125 GHz).

    Returns:
        np.ndarray: Filtered waveform, same shape as input.
    """
    if f3db_hz is None:
        f3db_hz = 0.5 * symbol_rate
    fs = symbol_rate * samples_per_ui
    b, a = bessel(order, f3db_hz, btype='low', analog=False, norm='mag', fs=fs)
    return lfilter(b, a, waveform)

def ffe(waveform, samples_per_ui, n_taps=5, tap_weights=None, sample_offset=None):
    """Applies a T-spaced feed-forward equalizer (FFE) per IEEE 802.3-2022 121.8.5.4.

    Downsamples to symbol rate at sample_offset, then applies a weighted sum over n_taps
    consecutive symbol-spaced samples. Default tap initialization places all weight on the
    center tap (identity: no equalization).

    Args:
        waveform (np.ndarray): Input waveform at N samples/UI (e.g. after rx_filter).
        samples_per_ui (int): Oversampling factor N.
        n_taps (int): Number of equalizer taps (5 per 121.8.5.4).
        tap_weights (array-like or None): Tap coefficients. None defaults to center-tap init.
        sample_offset (int or None): Sample index within each UI to use for downsampling.
                                     None defaults to samples_per_ui // 2 (center of UI).

    Returns:
        equalized (np.ndarray): Equalized symbol values, length len(sampled) - n_taps + 1.
        tap_weights (np.ndarray): The tap coefficients used.
    """
    if sample_offset is None:
        sample_offset = samples_per_ui // 2
    if tap_weights is None:
        tap_weights = np.zeros(n_taps)
        tap_weights[n_taps // 2] = 1.0
    else:
        tap_weights = np.asarray(tap_weights, dtype=float)

    sampled = waveform[sample_offset::samples_per_ui]
    n_out = len(sampled) - n_taps + 1
    equalized = np.array([np.dot(tap_weights, sampled[i:i + n_taps]) for i in range(n_out)])
    return equalized, tap_weights

def rx_threshold(samples, rx_power_levels):
    """Converts sampled values to PAM4 symbol decisions (0-3) using midpoint thresholds.

    Args:
        samples (array-like): Sampled values at symbol rate (e.g. FFE output).
        rx_power_levels (list of float): Four received power levels [R0, R1, R2, R3].

    Returns:
        np.ndarray: Symbol decisions in {0, 1, 2, 3}.
    """
    R0, R1, R2, R3 = rx_power_levels
    thresholds = [(R0 + R1) / 2, (R1 + R2) / 2, (R2 + R3) / 2]
    return np.digitize(np.asarray(samples), thresholds)

#%% Function testing


# Generating test patterns for lanes 0-3
seeds = [0b0000010101011, 0b0011101000001, 0b1001000101100, 0b0100010000010]  # Different seeds for each lane
test_patterns = []
for seed in seeds:
    prbs_bits = prbs_sequence(register_size=13, seed=seed)  # Generate full PRBS-13 sequence (8191 bits)
    prbs_bits += prbs_bits  # Duplicate the sequence to ensure even length for pairing
    gray_pairs = gray_code_pairs(prbs_bits)  # Pair and Gray-code the bits
    test_patterns.append(gray_pairs)

print("Generated PAM4 test patterns for lanes 0-3 (first 20 symbols of each lane):")
for lane, pattern in enumerate(test_patterns):
    print(f"Lane {lane}: {pattern[:20]}")

#%% Example usage: simulate the TX to RX pipeline for a single lane, first n_symbols symbols

symbol_rate = 26.5625e9 # 26.5625 GBd for 200GBASE-DR4
ui = 1 / symbol_rate
samples_per_ui = 16
transition_time = 10e-12
n_symbols = 50
tx_power_levels = [0.333, 0.667, 1.0, 1.333]
n_taps = 5
lane = 3

tx_pattern = test_patterns[lane][:n_symbols + n_taps + 3]
power = symbol_to_power(tx_pattern)
upsampled = upsample(power, samples_per_ui)
tx = tx_filter(upsampled, samples_per_ui, transition_time, ui)
loss_db, disp = fiber_loss_db(500), fiber_dispersion_ps_per_nm(500)
rx = channel_filter(tx, samples_per_ui, ui, loss_db=loss_db, dispersion_ps_per_nm=disp)
oe = rx_filter(rx, samples_per_ui, symbol_rate)
ffe_out, tap_w = ffe(oe, samples_per_ui, n_taps)

t = np.arange(n_symbols * samples_per_ui) * (ui / samples_per_ui) * 1e12  # ps
n_plot = n_symbols * samples_per_ui
ui_ps = ui * 1e12

rx_power_levels = [p * 10 ** (-loss_db / 10) for p in tx_power_levels]

n_ffe_plot = min(len(ffe_out), n_symbols - n_taps + 1) # Number of FFE output points to plot
rx_decisions = rx_threshold(ffe_out[:n_ffe_plot], rx_power_levels)

# Align the FFE output with the transmitted symbols by finding the offset that minimizes symbol errors

tx_syms = np.array(tx_pattern)

cursor = min(range(n_taps + 3),
             key=lambda d: np.sum(rx_decisions != tx_syms[d:d + n_ffe_plot]))

t_ffe = np.array([(i + cursor + 0.5) * ui_ps for i in range(n_ffe_plot)])
tx_syms_aligned = tx_syms[cursor:cursor + n_ffe_plot]
errors = np.where(rx_decisions != tx_syms_aligned)[0]
t_sym_tx = np.arange(n_symbols) * ui_ps
t_sym_rx = np.arange(n_ffe_plot) * ui_ps + cursor * ui_ps

# Set up a 3-panel plot: TX waveforms, RX waveforms, and TX vs RX symbol decisions

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1]})

for ax in axes:
    for k in range(n_symbols):
        if k % 2 == 0:
            ax.axvspan(k * ui_ps, (k + 1) * ui_ps, color='gray', alpha=0.15, lw=0)

# Panel 1: TX waveforms
ax1 = axes[0]
ax1.plot(t, upsampled[:n_plot], lw=0.8, label='Rectangular')
ax1.plot(t, tx[:n_plot], lw=0.8, label=f'After TX filter ({transition_time*1e12:.0f} ps rise time)')
for i, p in enumerate(tx_power_levels):
    ax1.axhline(p, color='red', lw=0.5, ls='--', alpha=0.5,
                label='TX power levels' if i == 0 else '_nolegend_')
ax1.set_ylabel('Power (mW)')
ax1.set_title('TX waveforms')
ax1.legend(loc='upper right', fontsize=8)

# Panel 2: RX waveforms
ax2 = axes[1]
ax2.plot(t, rx[:n_plot], lw=0.8, label=f'After channel ({loss_db:.2f} dB, {disp:.3f} ps/nm)')
ax2.plot(t, oe[:n_plot], lw=0.8, label='After RX filter (Bessel-Thomson)')
ax2.scatter(t_ffe, ffe_out[:n_ffe_plot], s=12, zorder=5, label='After FFE (center tap)')
for i, p in enumerate(rx_power_levels):
    ax2.axhline(p, color='blue', lw=0.5, ls='--', alpha=0.5,
                label='RX power levels' if i == 0 else '_nolegend_')
ax2.set_ylabel('Power (mW)')
ax2.set_title('RX waveforms')
ax2.legend(loc='upper right', fontsize=8)

# Panel 3: TX vs RX symbols
ax3 = axes[2]
ax3.step(t_sym_tx, tx_syms[:n_symbols], where='post', lw=1.2, label='TX symbols')
ax3.step(t_sym_rx, rx_decisions, where='post', lw=1.2, ls='--', label='RX decisions')
if len(errors) > 0:
    t_err = t_sym_rx[errors] + 0.5 * ui_ps
    ax3.scatter(t_err, rx_decisions[errors], marker='x', color='red', s=40, zorder=5,
                label=f'Errors ({len(errors)})')
ax3.set_yticks([0, 1, 2, 3])
ax3.set_ylabel('Symbol')
ax3.set_xlabel('Time (ps)')
ax3.set_title('TX vs RX symbol decisions')
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('waveforms.png', dpi=150)
plt.show()

# %%

