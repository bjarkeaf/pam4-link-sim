import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel

from sim import (
    prbs_sequence, gray_code_pairs,
    symbol_to_power, upsample,
    tx_filter, fiber_loss_db, channel_filter,
    rx_filter, ffe, ffe_ls, rx_threshold,
    compute_tdecq, plot_tdecq,
)

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
transition_time = 34e-12 # max allowed 20%-80% rise time per IEEE 802.3-2022 Table 121-6
n_symbols = 15
tx_power_levels = [0.333, 0.667, 1.0, 1.333]
n_taps = 5
lane = 0
# Approximate C_eq * sigma_G from TDECQ (hardcoded near the found value of ~0.018).
# Used to show the noise-limited waveform in the RX panel before TDECQ is computed.
noise_sigma = 0.02

tx_pattern = test_patterns[lane][:n_symbols + n_taps + 3]
power = symbol_to_power(tx_pattern)
upsampled = upsample(power, samples_per_ui)
tx = tx_filter(upsampled, samples_per_ui, transition_time, ui)
loss_db = fiber_loss_db(500)
disp = 0.8  # ps/nm, max positive dispersion per IEEE 802.3-2022 Table 121-13
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

# Re-run with LS-optimized taps now that cursor is known
ideal_levels_short = np.array([rx_power_levels[s] for s in tx_pattern[max(0, n_taps // 2 - cursor):]])
ffe_ls_out, tap_w_ls = ffe_ls(oe, ideal_levels_short, samples_per_ui)
n_ffe_plot = min(len(ffe_ls_out), n_symbols - n_taps + 1)
rx_decisions = rx_threshold(ffe_ls_out[:n_ffe_plot], rx_power_levels)

t_ffe = np.array([(i + cursor + 0.5) * ui_ps for i in range(n_ffe_plot)])
tx_syms_aligned = tx_syms[cursor:cursor + n_ffe_plot]
errors = np.where(rx_decisions != tx_syms_aligned)[0]
t_sym_tx = np.arange(n_symbols) * ui_ps
t_sym_rx = np.arange(n_ffe_plot) * ui_ps + cursor * ui_ps

# Set up a 3-panel plot: TX waveforms, RX waveforms, and TX vs RX symbol decisions
fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1]})

for ax in axes:
    for k in range(n_symbols):
        if k % 2 == 0:
            ax.axvspan(k * ui_ps, (k + 1) * ui_ps, color='gray', alpha=0.15, lw=0)

# Panel 1: TX waveforms
ax1 = axes[0]
ax1.plot(t, upsampled[:n_plot], lw=0.8, label='Rectangular')
ax1.plot(t, tx[:n_plot], lw=0.8, label=f'After TX filter')
for i, p in enumerate(tx_power_levels):
    ax1.axhline(p, color='red', lw=0.5, ls='--', alpha=0.5,
                label='TX power levels' if i == 0 else '_nolegend_')
ax1.set_ylim(bottom=0)
ax1.set_ylabel('Power (mW)')
ax1.set_title('TX waveforms')
ax1.legend(loc='lower right', fontsize=8)

# Panel 2: RX waveforms
ax2 = axes[1]
ax2.plot(t, rx[:n_plot], lw=0.8, label=f'After channel')
ax2.plot(t, oe[:n_plot], lw=0.8, label='After RX filter')
ax2.scatter(t_ffe, ffe_ls_out[:n_ffe_plot], s=12, zorder=5, label='After FFE')
rng = np.random.default_rng(42)
ffe_noisy = ffe_ls_out[:n_ffe_plot] + rng.normal(0, noise_sigma, n_ffe_plot)
#ax2.scatter(t_ffe, ffe_noisy, s=6, zorder=4, alpha=0.5, label=f'FFE + noise (σ={noise_sigma*1e3:.0f} mW)')
for i, p in enumerate(rx_power_levels):
    ax2.axhline(p, color='blue', lw=0.5, ls='--', alpha=0.5,
                label='RX power levels' if i == 0 else '_nolegend_')
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Power (mW)')
ax2.set_title('RX waveforms')
ax2.legend(loc='lower right', fontsize=8)

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
ax3.legend(loc='lower right', fontsize=8)

axes[0].set_xlim(0, n_symbols * ui_ps)

plt.tight_layout()
plt.savefig('waveforms.png', dpi=150)
plt.show()

#%% Demo: full TDECQ on PRBS13Q

# Run the full pipeline on the complete 8191-symbol pattern for the selected lane
tx_pattern_full = test_patterns[lane]
power_full     = symbol_to_power(tx_pattern_full)
upsampled_full = upsample(power_full, samples_per_ui)
tx_full  = tx_filter(upsampled_full, samples_per_ui, transition_time, ui)
rx_full  = channel_filter(tx_full, samples_per_ui, ui, loss_db=loss_db, dispersion_ps_per_nm=disp)
oe_full  = rx_filter(rx_full, samples_per_ui, symbol_rate)

# BT filter coefficients (needed for Ceq; matches what rx_filter uses internally)
b_bt, a_bt = bessel(4, 0.5 * symbol_rate, btype='low', analog=False, norm='mag',
                    fs=symbol_rate * samples_per_ui)

# Find cursor from the full pattern independently of the demo block above.
ffe_full_ct, _ = ffe(oe_full, samples_per_ui, n_taps)
n_ffe_full = len(ffe_full_ct)
rx_decisions_full = rx_threshold(ffe_full_ct, rx_power_levels)
tx_syms_full = np.array(tx_pattern_full)
cursor_full = min(range(n_taps + 3),
                  key=lambda d: np.sum(rx_decisions_full[:len(tx_syms_full) - d] != tx_syms_full[d:d + n_ffe_full]))
sym_offset = max(0, n_taps // 2 - cursor_full)
ideal_levels = np.array([rx_power_levels[s] for s in tx_pattern_full[sym_offset:]])

print("\nRunning TDECQ computation on full PRBS13Q pattern...")
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
