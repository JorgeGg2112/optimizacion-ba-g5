from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ba_optimizer import bat_algorithm, random_search
from problems_f5_e3 import sallen_key_butterworth_e3, schaffer_n2_shifted

def summarize_runs(runs: list[dict]) -> dict:
    finals = np.array([r["best_f"] for r in runs])
    times = np.array([r["elapsed_s"] for r in runs])
    return {
        "best": float(np.min(finals)),
        "median": float(np.median(finals)),
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals, ddof=1)),
        "worst": float(np.max(finals)),
        "avg_time_s": float(np.mean(times)),
        "finals": finals,
    }


def align_curves(runs: list[dict], max_evals: int = 3000):
    grid = np.arange(1, max_evals + 1)
    curves = np.empty((len(runs), max_evals), dtype=float)
    for idx, r in enumerate(runs):
        evals = r["hist_evals"]
        vals = r["hist_best"]
        pos = np.searchsorted(evals, grid, side="right") - 1
        pos = np.clip(pos, 0, len(vals) - 1)
        curves[idx] = vals[pos]
    return grid, curves


def run_benchmark(name: str, func, lb: np.ndarray, ub: np.ndarray, out_dir: Path, n_runs: int = 30):
    out_dir.mkdir(parents=True, exist_ok=True)
    max_evals = 3000

    ba_runs = [bat_algorithm(func, lb, ub, max_evals=max_evals, r0=0.5, seed=s) for s in range(1, n_runs + 1)]
    rs_runs = [random_search(func, lb, ub, max_evals=max_evals, seed=s) for s in range(1, n_runs + 1)]

    ba_stats = summarize_runs(ba_runs)
    rs_stats = summarize_runs(rs_runs)

    stat, p_val = wilcoxon(ba_stats["finals"], rs_stats["finals"], alternative="less")

    print(f"=== {name} ({n_runs} corridas, {max_evals} evals) ===")
    print(f"BA -> Mejor: {ba_stats['best']:.6e} | Mediana: {ba_stats['median']:.6e} | "
          f"Media: {ba_stats['mean']:.6e} | Std: {ba_stats['std']:.6e} | Peor: {ba_stats['worst']:.6e} | "
          f"Tiempo prom: {ba_stats['avg_time_s']*1000:.2f} ms")
    print(f"RS -> Mediana: {rs_stats['median']:.6e} | Media: {rs_stats['mean']:.6e}")
    print(f"Wilcoxon (BA < Random Search): W={stat:.4f}, p={p_val:.4e} ({'Significativo' if p_val < 0.05 else 'No significativo'})\n")

    grid, curves = align_curves(ba_runs, max_evals=max_evals)
    mean_c = np.mean(curves, axis=0)
    std_c = np.std(curves, axis=0, ddof=1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, mean_c, label="BA (media)", linewidth=2)
    plt.fill_between(grid, np.maximum(mean_c - std_c, 1e-12), mean_c + std_c, alpha=0.25, label="±1 std")
    plt.yscale("log")
    plt.xlabel("Evaluaciones de la función objetivo")
    plt.ylabel("Mejor f(x) encontrado")
    plt.title(f"Convergencia BA — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"convergencia_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()

    r0_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    box_data = []
    for r0_val in r0_levels:
        runs_r0 = [bat_algorithm(func, lb, ub, max_evals=max_evals, r0=r0_val, seed=s) for s in range(1, n_runs + 1)]
        box_data.append([r["best_f"] for r in runs_r0])

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(box_data, tick_labels=[str(v) for v in r0_levels])
    plt.yscale("log")
    plt.xlabel("Tasa inicial de emisión de pulsos (r0)")
    plt.ylabel("Valor final de f(x)")
    plt.title(f"Sensibilidad a r0 — {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"sensibilidad_r0_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    results_dir = Path("results")

    run_benchmark(
        "F5_SchafferN2_Desplazada",
        schaffer_n2_shifted,
        np.array([-100.0, -100.0]),
        np.array([100.0, 100.0]),
        results_dir,
        n_runs=30,
    )

    lb_e3 = np.array([3.0, 3.0, -9.0, -9.0, 3.0, 3.0, -9.0, -9.0])
    ub_e3 = np.array([6.0, 6.0, -6.0, -6.0, 6.0, 6.0, -6.0, -6.0])
    run_benchmark(
        "E3_SallenKey_Butterworth_BA",
        sallen_key_butterworth_e3,
        lb_e3,
        ub_e3,
        results_dir,
        n_runs=30,
    )
    