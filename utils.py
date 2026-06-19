#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, bessel, freqz

#%% PRBS and pattern generation

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

#%% Signal encoding and upsampling

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

#%% TX filter and fiber channel

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

#%% RX filter, FFE, and decision

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

def ffe_ls(waveform, ideal_levels, samples_per_ui, n_taps=5, sample_offset=None):
    """Finds optimal T-spaced FFE tap weights by least-squares MSE against ideal PAM4 levels.

    Builds the regression matrix X (each row: n_taps symbol-spaced samples at 1-UI spacing)
    and solves w = argmin ||Xw - d||^2. Tap weights are rescaled to sum = 1 per 121.8.5.4.

    Args:
        waveform (np.ndarray): Input waveform at N samples/UI (e.g. after rx_filter).
        ideal_levels (array-like): Ideal received PAM4 levels aligned with the FFE output.
        samples_per_ui (int): Oversampling factor N.
        n_taps (int): Number of equalizer taps (5 per 121.8.5.4).
        sample_offset (int or None): Phase within UI to sample. Defaults to center (spu // 2).

    Returns:
        equalized (np.ndarray): Equalized symbol values at symbol rate.
        tap_weights (np.ndarray): Optimized tap coefficients, summing to 1.
    """
    if sample_offset is None:
        sample_offset = samples_per_ui // 2

    sampled = waveform[sample_offset::samples_per_ui]
    n_out   = len(sampled) - n_taps + 1

    # Each row of X is a window of n_taps consecutive symbol-spaced samples
    X = np.array([sampled[i:i + n_taps] for i in range(n_out)])
    d = np.asarray(ideal_levels[:n_out], dtype=float)

    tap_weights, _, _, _ = np.linalg.lstsq(X, d, rcond=None)
    tap_weights /= tap_weights.sum()   # enforce sum = 1 (121.8.5.4)

    equalized = X @ tap_weights
    return equalized, tap_weights


def apply_ffe_fullrate(waveform, tap_weights, samples_per_ui):
    """Applies T-spaced FFE tap weights to a full-rate waveform, preserving sample rate.

    Computes y[i] = sum_k w_k * x[i + k * samples_per_ui], i.e. a sparse FIR filter
    with taps spaced 1 UI apart. This is needed to form the equalized eye diagram.

    Args:
        waveform (np.ndarray): Full-rate waveform (e.g. after rx_filter).
        tap_weights (np.ndarray): FFE tap coefficients (e.g. from ffe_ls).
        samples_per_ui (int): Oversampling factor N.

    Returns:
        np.ndarray: Equalized waveform, length len(waveform) - (n_taps - 1) * samples_per_ui.
    """
    n_taps = len(tap_weights)
    delay  = (n_taps - 1) * samples_per_ui
    n_out  = len(waveform) - delay
    out    = np.zeros(n_out)
    for k, w in enumerate(tap_weights):
        start  = k * samples_per_ui
        out   += w * waveform[start:start + n_out]
    return out


#%% Eye diagram and TDECQ measurement

def form_eye(waveform, samples_per_ui):
    """Folds a full-rate waveform into a 2D eye diagram array.

    Args:
        waveform (np.ndarray): Full-rate waveform.
        samples_per_ui (int): Oversampling factor N.

    Returns:
        np.ndarray: Shape (n_symbols, samples_per_ui). Each row is one UI period.
    """
    n_symbols = len(waveform) // samples_per_ui
    return waveform[:n_symbols * samples_per_ui].reshape(n_symbols, samples_per_ui)


def measure_eye(eye):
    """Estimates P_ave and OMAouter from a PAM4 eye diagram.

    P_ave is the mean of all samples. OMAouter is estimated as the difference between
    the top-quartile mean (P3 cluster) and the bottom-quartile mean (P0 cluster),
    which is accurate when all four PAM4 symbols are approximately equally likely.

    Args:
        eye (np.ndarray): Shape (n_symbols, samples_per_ui) from form_eye().

    Returns:
        dict with keys: p_ave, oma_outer, p_th1, p_th2, p_th3.
    """
    samples  = eye.flatten()
    p_ave     = float(np.mean(samples))

    sorted_s = np.sort(samples)
    n        = len(sorted_s)
    p0_est   = float(np.mean(sorted_s[:n // 4]))
    p3_est   = float(np.mean(sorted_s[3 * n // 4:]))
    oma_outer = p3_est - p0_est

    return {
        'p_ave':      p_ave,
        'oma_outer': oma_outer,
        'p_th1':      p_ave - oma_outer / 3,
        'p_th2':      p_ave,
        'p_th3':      p_ave + oma_outer / 3,
    }


def measure_oma_outer(ffe_waveform, tx_pattern, samples_per_ui, sym_offset):
    """Measures OMAouter per IEEE 802.3-2022 121.8.4.

    P0 = mean of the central 2 UI of the first run of >=6 consecutive 0s.
    P3 = mean of the central 2 UI of the first run of >=7 consecutive 3s.
    OMAouter = P3 - P0.

    Args:
        ffe_waveform (np.ndarray): Full-rate equalized waveform from apply_ffe_fullrate().
        tx_pattern (list[int]): Transmitted PAM4 symbol sequence.
        samples_per_ui (int): Oversampling factor.
        sym_offset (int): Number of symbols into tx_pattern that aligns with ffe_waveform[0].

    Returns:
        dict with keys: oma_outer, p0, p3.
    """
    def first_run(symbol, min_len):
        i = 0
        while i < len(tx_pattern):
            if tx_pattern[i] == symbol:
                j = i
                while j < len(tx_pattern) and tx_pattern[j] == symbol:
                    j += 1
                if j - i >= min_len:
                    return i, j - i
                i = j
            else:
                i += 1
        raise ValueError(f"No run of {min_len}+ symbol {symbol} found in pattern")

    def central_2ui_mean(run_start, run_len):
        center_offset = (run_len - 2) // 2      # offset of first central UI within run
        waveform_sym  = run_start - sym_offset + center_offset
        s = waveform_sym * samples_per_ui
        return float(np.mean(ffe_waveform[s : s + 2 * samples_per_ui]))

    p0_start, p0_len = first_run(0, 6)
    p3_start, p3_len = first_run(3, 7)
    p0 = central_2ui_mean(p0_start, p0_len)
    p3 = central_2ui_mean(p3_start, p3_len)
    return {'oma_outer': p3 - p0, 'p0': p0, 'p3': p3}


def compute_ceq(b, a, tap_weights, symbol_rate, samples_per_ui, n_fft=4096):
    """Computes the FFE noise enhancement factor Ceq (IEEE 802.3-2022 Eq. 121-9).

    Ceq = sqrt( integral N(f) * |H_eq(f)|^2 df )

    N(f) is white noise shaped by the BT filter response, normalized so integral N(f) df = 1.
    H_eq(f) is the T-spaced FFE frequency response at the same frequency grid.

    Args:
        b, a (array-like): BT filter coefficients from bessel().
        tap_weights (np.ndarray): FFE tap coefficients.
        symbol_rate (float): Symbol rate in Hz.
        samples_per_ui (int): Oversampling factor.
        n_fft (int): Number of frequency evaluation points.

    Returns:
        float: Ceq (dimensionless noise enhancement factor; 1.0 = no enhancement).
    """
    fs = symbol_rate * samples_per_ui
    freqs, H_bt = freqz(b, a, worN=n_fft // 2 + 1, fs=fs) # H_bt is the BT filter response at these frequencies

    N_f  = np.abs(H_bt) ** 2
    N_f /= np.trapezoid(N_f, freqs)              # normalize: integral N(f) df = 1

    # H_eq(f) = sum_k w_k * exp(-j 2pi f k T),  where T = 1 UI = 1/symbol_rate
    T    = 1.0 / symbol_rate
    k    = np.arange(len(tap_weights))
    H_eq = np.array([
        np.dot(tap_weights, np.exp(-1j * 2 * np.pi * f * k * T)) for f in freqs
    ])

    return float(np.sqrt(np.trapezoid(N_f * np.abs(H_eq) ** 2, freqs)))


def collect_histograms(eye, samples_per_ui, p_ave, oma_outer, n_bins=512):
    """Collects normalized vertical histograms at 0.45 UI and 0.55 UI (121.8.5.3).

    Each histogram is a probability mass function F(y_i) over optical power,
    where y_i are equally spaced bin centers spanning ±2 * OMAouter around P_ave.
    At samples_per_ui=16 the 0.04 UI window specified in the standard is smaller than
    one sample, so each histogram is collected from the single nearest sample column.

    Args:
        eye (np.ndarray): Shape (n_symbols, samples_per_ui) from form_eye().
        samples_per_ui (int): Oversampling factor N.
        p_ave (float): Mean optical power of the equalized eye.
        oma_outer (float): Outer OMA of the equalized eye.
        n_bins (int): Number of histogram bins.

    Returns:
        y_bins (np.ndarray): Bin centers, length n_bins.
        F_left (np.ndarray): Normalized histogram at 0.45 UI.
        F_right (np.ndarray): Normalized histogram at 0.55 UI.
    """
    phase_left  = round(0.45 * samples_per_ui)
    phase_right = round(0.55 * samples_per_ui)

    bin_edges = np.linspace(p_ave - 2 * oma_outer, p_ave + 2 * oma_outer, n_bins + 1)
    y_bins    = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _histogram(col):
        counts, _ = np.histogram(eye[:, col], bins=bin_edges)
        return counts / counts.sum()

    return y_bins, _histogram(phase_left), _histogram(phase_right)


def _cdf_between(F, y_bins, pth):
    """Probability mass between each bin y_i and threshold pth (IEEE 802.3-2022 Eq. 121-4).

    CDF[i] = sum of F(y) for y between y_i and pth (inclusive on both ends).
    Accumulation always runs from the threshold outward toward y_i, so CDF grows
    as y_i moves away from the threshold.

    For y_i >= pth:  CDF[i] = sum F(y) for y in [pth, y_i]   (upward from threshold)
    For y_i <  pth:  CDF[i] = sum F(y) for y in [y_i, pth]   (downward from threshold)
    """
    ith = np.searchsorted(y_bins, pth)      # index of bin nearest to threshold
    cdf = np.zeros_like(F)

    # Above threshold: cumulative sum going upward from threshold
    cdf[ith:] = np.cumsum(F[ith:])

    # Below threshold: cumulative sum going downward from threshold
    # cumsum(F[ith::-1])[j] = sum(F[ith-j : ith+1]), so cdf[i] = cumsum[ith-i]
    if ith > 0:
        cum_down  = np.cumsum(F[ith::-1])   # [F[ith], F[ith]+F[ith-1], ...]
        cdf[:ith] = cum_down[ith:0:-1]      # reorder: cdf[i] = sum(F[i : ith+1])

    return cdf


def compute_ser(y_bins, F_hist, p_ave, oma_outer, sigma_g, c_eq):
    """Estimates PAM4 SER for one histogram at a given noise level sigma_g.

    For each of the three sub-eye thresholds, convolves the CDF (probability mass
    between y_i and the threshold) with a Gaussian kernel of std c_eq * sigma_g
    (IEEE 802.3-2022 Eq. 121-5 to 121-8). Returns the sum of three partial SERs.

    Args:
        y_bins (np.ndarray): Bin centers from collect_histograms().
        F_hist (np.ndarray): Normalized histogram (sums to 1).
        p_ave (float): Mean power of the equalized eye.
        oma_outer (float): Outer OMA.
        sigma_g (float): Swept noise level (RMS).
        c_eq (float): FFE noise enhancement factor.

    Returns:
        float: Estimated SER for this histogram.
    """
    p_th1 = p_ave - oma_outer / 3
    p_th2 = p_ave
    p_th3 = p_ave + oma_outer / 3

    delta_y   = y_bins[1] - y_bins[0]
    sigma_eff = c_eq * sigma_g            # effective noise std at the decision threshold

    ser = 0.0
    for pth in [p_th1, p_th2, p_th3]:
        cdf    = _cdf_between(F_hist, y_bins, pth)
        kernel = (delta_y / (sigma_eff * np.sqrt(2 * np.pi))) * \
                 np.exp(-0.5 * ((y_bins - pth) / sigma_eff) ** 2)
        ser   += np.dot(cdf, kernel)

    return ser


def compute_tdecq(waveform, ideal_levels, b_bt, a_bt, samples_per_ui, symbol_rate,
                  sigma_s=0.0, n_bins=512, tol=1e-5, max_iter=60,
                  tx_pattern=None, sym_offset=0):
    """Computes TDECQ per IEEE 802.3-2022 Clause 121.8.5.3.

    Pipeline:
      1. Find optimal FFE taps by least-squares against ideal_levels.
      2. Apply FFE at full sample rate to form the equalized eye diagram.
      3. Measure P_ave, OMAouter, and collect histograms at 0.45 / 0.55 UI.
      4. Bisect sigma_G until max(SER_L, SER_R) = 4.8e-4 (target SER).
      5. Compute TDECQ = 10 log10(OMAouter / (6 * Qt * R)).

    Args:
        waveform (np.ndarray): BT-filtered waveform at full sample rate.
        ideal_levels (array-like): Ideal received PAM4 power levels aligned with the
                                   FFE output (symbol index 0 = first equalized symbol).
        b_bt, a_bt: BT filter coefficients from bessel(), used to compute Ceq.
        samples_per_ui (int): Oversampling factor.
        symbol_rate (float): Symbol rate in Hz.
        sigma_s (float): Instrument noise RMS (= 0 for software simulation).
        n_bins (int): Number of histogram bins.
        tol (float): Fractional SER convergence tolerance.
        max_iter (int): Maximum bisection iterations.

    Returns:
        dict: tdecq_db, sigma_g, R, c_eq, tap_weights, oma_outer, p_ave, ser,
              y_bins, F_left, F_right, eye.
    """
    TARGET_SER = 4.8e-4

    # Optimal FFE taps by least-squares MSE (tap weights are reused for all sigma_g values)
    _, tap_weights = ffe_ls(waveform, ideal_levels, samples_per_ui)

    # Apply FFE at full sample rate to preserve the eye diagram structure
    ffe_waveform = apply_ffe_fullrate(waveform, tap_weights, samples_per_ui)

    # Eye diagram and signal levels
    eye   = form_eye(ffe_waveform, samples_per_ui)
    p_ave = float(np.mean(eye))
    if tx_pattern is not None:
        # Spec-accurate OMAouter: mean of settled levels P0 and P3 (121.8.4)
        om = measure_oma_outer(ffe_waveform, tx_pattern, samples_per_ui, sym_offset)
        oma_outer = om['oma_outer']
    else:
        oma_outer = measure_eye(eye)['oma_outer']

    # Histograms at 0.45 and 0.55 UI (computed once; fixed for all sigma_g iterations)
    y_bins, F_left, F_right = collect_histograms(eye, samples_per_ui, p_ave, oma_outer, n_bins)

    # FFE noise enhancement factor (fixed for these tap weights)
    c_eq = compute_ceq(b_bt, a_bt, tap_weights, symbol_rate, samples_per_ui)

    # Bisect sigma_g: find the largest noise the eye can absorb at SER = TARGET_SER
    sigma_g_lo, sigma_g_hi = 1e-6 * oma_outer, oma_outer
    sigma_g = sigma_g_lo
    ser     = 1.0
    for _ in range(max_iter):
        sigma_g = 0.5 * (sigma_g_lo + sigma_g_hi)
        ser_l   = compute_ser(y_bins, F_left,  p_ave, oma_outer, sigma_g, c_eq)
        ser_r   = compute_ser(y_bins, F_right, p_ave, oma_outer, sigma_g, c_eq)
        ser     = max(ser_l, ser_r)
        if abs(ser - TARGET_SER) / TARGET_SER < tol:
            break
        if ser > TARGET_SER:
            sigma_g_hi = sigma_g   # eye cannot absorb this much noise -- reduce
        else:
            sigma_g_lo = sigma_g   # room for more noise -- increase

    R        = np.sqrt(sigma_g ** 2 + sigma_s ** 2)
    Qt       = 3.414               # Q-factor for SER = 4.8e-4, Gray-coded PAM4 (Table 121-12)
    tdecq_db = 10 * np.log10(oma_outer / (6 * Qt * R))

    return {
        'tdecq_db':    tdecq_db,
        'sigma_g':     sigma_g,
        'R':           R,
        'c_eq':        c_eq,
        'tap_weights': tap_weights,
        'oma_outer':   oma_outer,
        'p_ave':        p_ave,
        'ser':         ser,
        'y_bins':      y_bins,
        'F_left':      F_left,
        'F_right':     F_right,
        'eye':         eye,
    }


#%% Visualization

def plot_tdecq(result, samples_per_ui, ui, save_path='tdecq.png'):
    """Plots a 2-panel TDECQ summary figure and saves to disk.

    Panel 1: Equalized eye diagram (density plot) with P_th1/2/3, P_ave, and histogram
             window positions at 0.45 / 0.55 UI marked.
    Panel 2: Left and right vertical histograms with Gaussian kernels at converged sigma_g.

    Args:
        result (dict): Output from compute_tdecq().
        samples_per_ui (int): Oversampling factor N.
        ui (float): Symbol period in seconds.
        save_path (str): File path for the saved figure.
    """
    p_ave, oma_outer = result['p_ave'], result['oma_outer']
    p_th1 = p_ave - oma_outer / 3
    p_th2 = p_ave
    p_th3 = p_ave + oma_outer / 3
    eye  = result['eye']
    ui_ps = ui * 1e12

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Panel 1: Eye diagram as 2D density ---
    ax1 = axes[0]
    t_col = np.linspace(0, ui_ps, samples_per_ui, endpoint=False)
    t_eye = np.tile(t_col, len(eye))          # same time positions for every row
    p_eye = eye.flatten()
    counts, t_edges, p_edges = np.histogram2d(
        t_eye, p_eye, bins=[samples_per_ui, 200],
        range=[[0, ui_ps], [p_eye.min(), p_eye.max()]]
    )
    ax1.pcolormesh(t_edges, p_edges, counts.T, cmap='hot_r')
    # Horizontal threshold lines with inline labels on the left edge
    for p, label in [(p_th1, '$P_{th1}$'), (p_th2, '$P_{th2}=P_{ave}$'), (p_th3, '$P_{th3}$')]:
        ax1.axhline(p, color='grey', lw=0.8, ls='--', alpha=0.7)
        ax1.text(0.01, p, label, transform=ax1.get_yaxis_transform(),
                 va='bottom', ha='left', fontsize=9, color='grey')
    # Vertical histogram window lines with inline labels inside the top of the plot
    for phase_frac, label in [(0.45, '0.45 UI'), (0.55, '0.55 UI')]:
        ax1.axvline(phase_frac * ui_ps, color='grey', lw=1, ls=':', alpha=0.7)
        ax1.text(phase_frac * ui_ps, 0.97, label, transform=ax1.get_xaxis_transform(),
                 va='top', ha='center', fontsize=9, color='grey')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Power (mW)')
    ax1.set_title('Equalized eye diagram')

    # --- Panel 2: Histograms with Gaussian kernels ---
    ax2 = axes[1]
    y_bins = result['y_bins']
    delta_y = y_bins[1] - y_bins[0]
    ax2.barh(y_bins, result['F_left'],  height=delta_y, alpha=0.5, label='0.45 UI')
    ax2.barh(y_bins, result['F_right'], height=delta_y, alpha=0.5, label='0.55 UI')
    sigma_eff = result['c_eq'] * result['sigma_g']
    scale = max(result['F_left'].max(), result['F_right'].max())
    for i, pth in enumerate([p_th1, p_th2, p_th3]):
        kernel = np.exp(-0.5 * ((y_bins - pth) / sigma_eff) ** 2)
        ax2.plot(kernel * scale, y_bins, 'k--', lw=0.8, alpha=0.6,
                 label=f'Gaussian kernel ($C_{{eq}}\\sigma_G$={sigma_eff*1e3:.1f} mW)' if i == 0 else '_nolegend_')
        ax2.axhline(pth, color='red', lw=0.8, ls='--', alpha=0.4,
                    label='Decision thresholds' if i == 0 else '_nolegend_')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel('Probability mass')
    ax2.set_ylabel('Power (mW)')
    ax2.set_title('Histograms at 0.45 / 0.55 UI')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")


#%% Demo: 50-symbol pipeline


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

symbol_rate = 26.5625e9 # 26.5625 GBd for 200GBASE-DR4
ui = 1 / symbol_rate
samples_per_ui = 100
transition_time = 10e-12
n_symbols = 50
tx_power_levels = [0.333, 0.667, 1.0, 1.333]
n_taps = 5
lane = 3
# Approximate C_eq * sigma_G from TDECQ (hardcoded near the found value of ~0.018).
# Used to show the noise-limited waveform in the RX panel before TDECQ is computed.
noise_sigma = 0.02

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
ax1.set_ylim(bottom=0)
ax1.set_ylabel('Power (mW)')
ax1.set_title('TX waveforms')
ax1.legend(loc='upper right', fontsize=8)

# Panel 2: RX waveforms
ax2 = axes[1]
ax2.plot(t, rx[:n_plot], lw=0.8, label=f'After channel ({loss_db:.2f} dB, {disp:.3f} ps/nm)')
ax2.plot(t, oe[:n_plot], lw=0.8, label='After RX filter (Bessel-Thomson)')
ax2.scatter(t_ffe, ffe_out[:n_ffe_plot], s=12, zorder=5, label='After FFE (center tap)')
rng = np.random.default_rng(42)
ffe_noisy = ffe_out[:n_ffe_plot] + rng.normal(0, noise_sigma, n_ffe_plot)
ax2.scatter(t_ffe, ffe_noisy, s=6, zorder=4, alpha=0.5, label=f'FFE + noise (σ={noise_sigma*1e3:.0f} mW)')
for i, p in enumerate(rx_power_levels):
    ax2.axhline(p, color='blue', lw=0.5, ls='--', alpha=0.5,
                label='RX power levels' if i == 0 else '_nolegend_')
ax2.set_ylim(bottom=0)
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

#%% Demo: full TDECQ on PRBS13Q

# Run the full pipeline on the complete 8191-symbol pattern for the selected lane
tx_pattern_full = test_patterns[lane]
power_full    = symbol_to_power(tx_pattern_full)
upsampled_full = upsample(power_full, samples_per_ui)
tx_full  = tx_filter(upsampled_full, samples_per_ui, transition_time, ui)
rx_full  = channel_filter(tx_full, samples_per_ui, ui, loss_db=loss_db, dispersion_ps_per_nm=disp)
oe_full  = rx_filter(rx_full, samples_per_ui, symbol_rate)

# BT filter coefficients (needed for Ceq; matches what rx_filter uses internally)
b_bt, a_bt = bessel(4, 0.5 * symbol_rate, btype='low', analog=False, norm='mag',
                    fs=symbol_rate * samples_per_ui)

# Align ideal levels with the FFE output: the dominant tap is at n_taps//2 (center),
# but the waveform is already delayed by `cursor` symbols, so net offset = n_taps//2 - cursor.
ideal_levels = np.array([rx_power_levels[s] for s in tx_pattern_full[max(0, n_taps // 2 - cursor):]])

print("\nRunning TDECQ computation on full PRBS13Q pattern...")
sym_offset = max(0, n_taps // 2 - cursor)
tdecq_result = compute_tdecq(oe_full, ideal_levels, b_bt, a_bt, samples_per_ui, symbol_rate,
                             tx_pattern=tx_pattern_full, sym_offset=sym_offset)

print(f"TDECQ:     {tdecq_result['tdecq_db']:+.3f} dB")
print(f"sigma_G:   {tdecq_result['sigma_g']*1e3:.4f} mW")
print(f"C_eq:      {tdecq_result['c_eq']:.4f}")
print(f"OMAouter:  {tdecq_result['oma_outer']*1e3:.3f} mW")
print(f"P_ave:     {tdecq_result['p_ave']*1e3:.3f} mW")
print(f"SER (predicted, at sigma_G): {tdecq_result['ser']:.2e}")
print(f"Tap weights: {np.round(tdecq_result['tap_weights'], 4)}")

# Actual SER: decode the equalized waveform symbol by symbol and compare to transmitted pattern.
# The center column of the eye (phase = spu//2) gives the same samples as ffe_ls at its
# default sample_offset, so alignment with ideal_levels (and thus tx_pattern_full) is exact.
p_th1 = tdecq_result['p_ave'] - tdecq_result['oma_outer'] / 3
p_th2 = tdecq_result['p_ave']
p_th3 = tdecq_result['p_ave'] + tdecq_result['oma_outer'] / 3
equalized_sr = tdecq_result['eye'][:, samples_per_ui // 2]
decoded = np.where(equalized_sr >= p_th3, 3,
          np.where(equalized_sr >= p_th2, 2,
          np.where(equalized_sr >= p_th1, 1, 0)))
ref_symbols = np.array(tx_pattern_full[sym_offset : sym_offset + len(decoded)])
n_errors = int(np.sum(decoded != ref_symbols))
print(f"SER (actual, no added noise): {n_errors / len(decoded):.2e}  ({n_errors} errors / {len(decoded)} symbols)")

plot_tdecq(tdecq_result, samples_per_ui, ui)

# %%

