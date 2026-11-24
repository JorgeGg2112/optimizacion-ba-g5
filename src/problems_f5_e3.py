import numpy as np

E12_MANTISAS = np.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2])
E24_MANTISAS = np.array([
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
])


def snap_commercial_log10(log_val: float, mantisas: np.ndarray) -> float:
    decade = np.floor(log_val)
    candidates = decade + np.log10(mantisas)
    idx = int(np.argmin(np.abs(candidates - log_val)))
    return float(10.0 ** candidates[idx])


def schaffer_n2_shifted(x: np.ndarray, shift: np.ndarray = np.array([23.73, -26.60])) -> float:
    u = float(x[0] - shift[0])
    v = float(x[1] - shift[1])
    num = (np.sin(u * u - v * v) ** 2) - 0.5
    den = (1.0 + 0.001 * (u * u + v * v)) ** 2
    return 0.5 + num / den


def sallen_key_butterworth_e3(x: np.ndarray, fc: float = 1000.0) -> float:
    w_c = 2.0 * np.pi * fc
    q_targets = [0.5412, 1.3066]
    total_loss = 0.0

    for stage in range(2):
        base = 4 * stage
        r1 = snap_commercial_log10(float(x[base + 0]), E24_MANTISAS)
        r2 = snap_commercial_log10(float(x[base + 1]), E24_MANTISAS)
        c1 = snap_commercial_log10(float(x[base + 2]), E12_MANTISAS)
        c2 = snap_commercial_log10(float(x[base + 3]), E12_MANTISAS)

        prod_rc = r1 * r2 * c1 * c2
        w0 = 1.0 / np.sqrt(prod_rc)
        q = np.sqrt(prod_rc) / (c2 * (r1 + r2))

        total_loss += abs(w0 - w_c) / w_c + abs(q - q_targets[stage]) / q_targets[stage]

    return float(total_loss)


def trace_first_iteration_f5():
    s = np.array([23.73, -26.60])
    lb, ub = np.array([-100.0, -100.0]), np.array([100.0, 100.0])
    X = np.array([
        [10.80, 98.03],
        [30.76, 21.99],
        [-32.14, 93.65],
    ])
    V = np.zeros_like(X)
    f_min, f_max = 0.0, 2.0
    A0, r0, alpha, gamma = 1.0, 0.5, 0.9, 0.9

    U = np.array([
        [0.14, 0.10, 0.71, 0.84, 0.09, 0.47, 0.15, 0.30, 0.07, 0.96, 0.05, 0.19, 0.06, 0.27, 0.56, 0.02, 0.14, 0.61, 0.77, 0.78],
        [0.74, 0.44, 0.69, 0.74, 0.33, 0.47, 0.21, 0.95, 0.83, 0.02, 0.49, 0.97, 0.44, 0.15, 0.46, 0.39, 0.34, 0.22, 0.37, 0.31],
        [0.43, 0.68, 0.74, 0.30, 0.76, 0.21, 0.81, 0.94, 0.01, 0.62, 0.51, 0.18, 0.46, 0.75, 0.95, 0.59, 0.58, 0.13, 0.33, 0.63],
    ])

    fitness = np.array([schaffer_n2_shifted(ind, s) for ind in X])
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    A = np.full(3, A0)
    r = np.full(3, r0)
    mean_A = float(np.mean(A))

    print("Fitness inicial:", fitness, "| Mejor inicial:", best_idx + 1, best_f)
    for i in range(3):
        beta = U[i, 0]
        freq = f_min + (f_max - f_min) * beta
        V[i] = V[i] + (X[i] - best_x) * freq
        sol = np.clip(X[i] + V[i], lb, ub)

        u_local = U[i, 1]
        if u_local > r[i]:
            eps = np.array([2.0 * U[i, 2] - 1.0, 2.0 * U[i, 3] - 1.0])
            sol = np.clip(best_x + eps * mean_A, lb, ub)

        f_new = schaffer_n2_shifted(sol, s)
        u_acc = U[i, 4]
        accepted = (f_new <= fitness[i]) and (u_acc < A[i])
        if accepted:
            X[i] = sol
            fitness[i] = f_new
            A[i] = alpha * A[i]
            r[i] = r0 * (1.0 - np.exp(-gamma * 1))
        if f_new < best_f:
            best_f = f_new
            best_x = sol.copy()

        print(f"Individuo {i+1}: freq={freq:.4f}, sol={np.round(sol, 4)}, f_new={f_new:.6f}, aceptado={accepted}")


if __name__ == "__main__":
    trace_first_iteration_f5()