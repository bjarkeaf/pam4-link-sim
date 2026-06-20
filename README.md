# pam4-link-sim

PAM4 optical link simulator for IEEE 802.3-2022 Clause 121 (200GBASE-DR4). Models the TX-fiber-RX pipeline and computes TDECQ per the spec measurement procedure.

## Run

```bash
pip install numpy scipy matplotlib
python demo.py
```

## Outputs

- `waveforms.png`: TX/RX waveforms and symbol decisions for a short pattern
- `tdecq.png`: equalized eye diagram and histograms with converged TDECQ value
- Terminal: TDECQ (dB), OMAouter, sigma_G, C_eq, tap weights, SER
