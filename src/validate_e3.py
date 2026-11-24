from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from ba_optimizer import bat_algorithm
from problems_f5_e3 import (
    E12_MANTISAS,
    E24_MANTISAS,
    sallen_key_butterworth_e3,
    snap_commercial_log10,
)


def decode_e3_components(x: np.ndarray) -> dict:
    vals = []
    for stage in range(2):
        b = 4 * stage
        r1 = snap_commercial_log10(float(x[b + 0]), E24_MANTISAS)
        r2 = snap_commercial_log10(float(x[b + 1]), E24_MANTISAS)
        c1 = snap_commercial_log10(float(x[b + 2]), E12_MANTISAS)
        c2 = snap_commercial_log10(float(x[b + 3]), E12_MANTISAS)
        vals.append((r1, r2, c1, c2))
    return {"stage1": vals[0], "stage2": vals[1]}


def sallen_key_stage_tf(s: np.ndarray, r1: float, r2: float, c1: float, c2: float, gbw_hz: float | None = None) -> np.ndarray:
    w0_sq = 1.0 / (r1 * r2 * c1 * c2)
    a1 = c2 * (r1 + r2)
    if gbw_hz is None:
        return w0_sq / ((s ** 2) + a1 * w0_sq * s + w0_sq)

    wa = 2.0 * np.pi * gbw_hz
    g_amp = wa / (s + wa)
    damp = (1.0 / (r1 * c1)) + (1.0 / (r2 * c1)) + ((1.0 - g_amp) / (r2 * c2))
    return (g_amp * w0_sq) / ((s ** 2) + damp * s + w0_sq)


def cascade_mag_db(freqs_hz: np.ndarray, comps: dict, gbw_hz: float | None = None) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs_hz
    h1 = sallen_key_stage_tf(s, *comps["stage1"], gbw_hz=gbw_hz)
    h2 = sallen_key_stage_tf(s, *comps["stage2"], gbw_hz=gbw_hz)
    mag = np.abs(h1 * h2)
    return 20.0 * np.log10(np.maximum(mag, 1e-15))


def find_f_3db(comps: dict, gbw_hz: float | None = None) -> float:
    func = lambda f: float(cascade_mag_db(np.array([f]), comps, gbw_hz=gbw_hz)[0]) + 3.0
    return float(brentq(func, 100.0, 10000.0))


def export_spice_netlist(comps: dict, filepath: Path):
    r1_1, r2_1, c1_1, c2_1 = comps["stage1"]
    r1_2, r2_2, c1_2, c2_2 = comps["stage2"]
    netlist = f"""* Filtro Butterworth 4to Orden Sallen-Key (fc = 1 kHz) - Grupo 5
Vin in 0 AC 1

* Etapa 1 (Q1* = 0.5412)
R1_1 in n1 {r1_1:.6g}
R2_1 n1 n2 {r2_1:.6g}
C1_1 n1 out1 {c1_1:.6e}
C2_1 n2 0 {c2_1:.6e}
E1 out1 0 n2 out1 1e6

* Etapa 2 (Q2* = 1.3066)
R1_2 out1 n3 {r1_2:.6g}
R2_2 n3 n4 {r2_2:.6g}
C1_2 n3 out {c1_2:.6e}
C2_2 n4 0 {c2_2:.6e}
E2 out 0 n4 out 1e6

.ac dec 200 10 100k
.end
"""
    filepath.write_text(netlist, encoding="utf-8")


def run_validations_e3():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    lb = np.array([3.0, 3.0, -9.0, -9.0, 3.0, 3.0, -9.0, -9.0])
    ub = np.array([6.0, 6.0, -6.0, -6.0, 6.0, 6.0, -6.0, -6.0])
    runs = [bat_algorithm(sallen_key_butterworth_e3, lb, ub, max_evals=3000, r0=0.5, seed=s) for s in range(1, 31)]
    best_run = min(runs, key=lambda r: r["best_f"])
    comps = decode_e3_components(best_run["best_x"])

    print("=== MEJOR DISEÑO COMERCIAL ENCONTRADO POR BA (E3) ===")
    print(f"Valor función objetivo f(x): {best_run['best_f']:.6f}")
    for idx, key in enumerate(["stage1", "stage2"], start=1):
        r1, r2, c1, c2 = comps[key]
        w0 = 1.0 / np.sqrt(r1 * r2 * c1 * c2)
        q = np.sqrt(r1 * r2 * c1 * c2) / (c2 * (r1 + r2))
        print(f"Etapa {idx}: R1={r1:g} Ω, R2={r2:g} Ω, C1={c1*1e9:g} nF, C2={c2*1e9:g} nF | f0={w0/(2*np.pi):.2f} Hz, Q={q:.4f}")

    freqs = np.logspace(1, 5, 1000)
    fc = 1000.0
    mag_ideal_bw = 20.0 * np.log10(1.0 / np.sqrt(1.0 + (freqs / fc) ** 8))
    mag_design_ideal_opamp = cascade_mag_db(freqs, comps, gbw_hz=None)

    passband_mask = freqs <= fc
    max_err_pb = float(np.max(np.abs(mag_design_ideal_opamp[passband_mask] - mag_ideal_bw[passband_mask])))
    f_3db = find_f_3db(comps, gbw_hz=None)
    att_2fc, att_10fc = cascade_mag_db(np.array([2000.0, 10000.0]), comps, gbw_hz=None)

    print("\n--- VALIDACIÓN V1 (Respuesta en frecuencia) ---")
    print(f"Frecuencia real -3 dB: {f_3db:.2f} Hz (ideal: 1000.00 Hz)")
    print(f"Error máximo en banda de paso (<= 1 kHz): {max_err_pb:.4f} dB")
    print(f"Atenuación a 2fc (2 kHz): {att_2fc:.2f} dB (ideal: -24.08 dB)")
    print(f"Atenuación a 10fc (10 kHz): {att_10fc:.2f} dB (ideal: -80.00 dB)")

    mag_tl072 = cascade_mag_db(freqs, comps, gbw_hz=3.0e6)
    mag_lm358 = cascade_mag_db(freqs, comps, gbw_hz=1.0e6)
    f3_tl072 = find_f_3db(comps, gbw_hz=3.0e6)
    f3_lm358 = find_f_3db(comps, gbw_hz=1.0e6)
    export_spice_netlist(comps, out_dir / "filtro_butterworth_e3.cir")

    print("\n--- VALIDACIÓN V2 (Efecto GBW Op-Amp real y netlist SPICE) ---")
    print(f"f_-3dB Op-Amp Ideal: {f_3db:.2f} Hz | TL072 (3 MHz): {f3_tl072:.2f} Hz | LM358 (1 MHz): {f3_lm358:.2f} Hz")
    print(f"Netlist SPICE exportado en: {out_dir / 'filtro_butterworth_e3.cir'}")

    plt.figure(figsize=(8.5, 4.8))
    plt.semilogx(freqs, mag_ideal_bw, "k--", label="Butterworth Teórico Ideal", linewidth=1.8)
    plt.semilogx(freqs, mag_design_ideal_opamp, label="Diseño Comercial (Op-Amp Ideal)", linewidth=2)
    plt.semilogx(freqs, mag_tl072, label="Diseño Comercial + TL072 (GBW=3 MHz)", alpha=0.85)
    plt.semilogx(freqs, mag_lm358, label="Diseño Comercial + LM358 (GBW=1 MHz)", alpha=0.85)
    plt.axvline(1000.0, color="gray", linestyle=":", linewidth=1)
    plt.axhline(-3.0, color="gray", linestyle=":", linewidth=1)
    plt.ylim(-100, 5)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud |H(jf)| (dB)")
    plt.title("Validación V1 y V2 — Respuesta en Frecuencia Sallen-Key 4.º Orden")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "validacion_v1_v2_bode.png", dpi=200)
    plt.close()

    rng = np.random.default_rng(42)
    n_mc = 500
    fc_samples = np.empty(n_mc)
    q2_samples = np.empty(n_mc)

    r1_1, r2_1, c1_1, c2_1 = comps["stage1"]
    r1_2, r2_2, c1_2, c2_2 = comps["stage2"]

    for k in range(n_mc):
        tol_r = rng.uniform(0.95, 1.05, size=4)
        tol_c = rng.uniform(0.90, 1.10, size=4)
        mc_comps = {
            "stage1": (r1_1 * tol_r[0], r2_1 * tol_r[1], c1_1 * tol_c[0], c2_1 * tol_c[1]),
            "stage2": (r1_2 * tol_r[2], r2_2 * tol_r[3], c1_2 * tol_c[2], c2_2 * tol_c[3]),
        }
        fc_samples[k] = find_f_3db(mc_comps, gbw_hz=None)
        ra, rb, ca, cb = mc_comps["stage2"]
        q2_samples[k] = np.sqrt(ra * rb * ca * cb) / (cb * (ra + rb))

    in_spec_pct = 100.0 * np.mean((fc_samples >= 950.0) & (fc_samples <= 1050.0))
    print("\n--- VALIDACIÓN V3 (Monte Carlo 500 realizaciones: R ±5%, C ±10%) ---")
    print(f"fc media: {np.mean(fc_samples):.2f} Hz ± {np.std(fc_samples):.2f} Hz")
    print(f"Q2 media: {np.mean(q2_samples):.4f} ± {np.std(q2_samples):.4f} (objetivo: 1.3066)")
    print(f"Circuitos con fc dentro de ±5% [950, 1050] Hz: {in_spec_pct:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(fc_samples, bins=25, color="#2b6cb0", edgecolor="white")
    axes[0].axvline(1000.0, color="red", linestyle="--", label="Nominal (1000 Hz)")
    axes[0].axvspan(950.0, 1050.0, color="green", alpha=0.15, label="Banda ±5%")
    axes[0].set_title("Distribución de fc (-3 dB)")
    axes[0].set_xlabel("Frecuencia de corte (Hz)")
    axes[0].set_ylabel("Frecuencia (circuitos)")
    axes[0].legend()

    axes[1].hist(q2_samples, bins=25, color="#2f855a", edgecolor="white")
    axes[1].axvline(1.3066, color="red", linestyle="--", label="Q2* ideal (1.3066)")
    axes[1].set_title("Distribución de Q2 (Etapa 2)")
    axes[1].set_xlabel("Factor de calidad Q2")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "validacion_v3_montecarlo.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    run_validations_e3()