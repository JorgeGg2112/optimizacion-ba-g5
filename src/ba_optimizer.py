import time
import numpy as np


class EvalCounter:
    def __init__(self, max_evals: int = 3000):
        self.max_evals = max_evals
        self.count = 0
        self.best_f = np.inf
        self.history_evals = []
        self.history_best = []

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.count)

    def eval(self, func, x: np.ndarray) -> float:
        if self.count >= self.max_evals:
            return self.best_f
        val = float(func(x))
        self.count += 1
        if val < self.best_f:
            self.best_f = val
        self.history_evals.append(self.count)
        self.history_best.append(self.best_f)
        return val


def bat_algorithm(
    func,
    lb: np.ndarray,
    ub: np.ndarray,
    n_pop: int = 30,
    max_evals: int = 3000,
    f_min: float = 0.0,
    f_max: float = 2.0,
    A0: float = 1.0,
    r0: float = 0.5,
    alpha: float = 0.9,
    gamma: float = 0.9,
    seed: int = 1,
):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size

    counter = EvalCounter(max_evals=max_evals)

    X = rng.uniform(lb, ub, size=(n_pop, dim))
    V = np.zeros((n_pop, dim), dtype=float)
    A = np.full(n_pop, A0, dtype=float)
    r = np.full(n_pop, r0, dtype=float)

    fitness = np.empty(n_pop, dtype=float)
    for i in range(n_pop):
        if counter.remaining == 0:
            break
        fitness[i] = counter.eval(func, X[i])

    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    t_start = time.perf_counter()
    t = 0

    while counter.remaining > 0:
        t += 1
        mean_A = float(np.mean(A))

        for i in range(n_pop):
            if counter.remaining == 0:
                break

            beta = rng.random()
            freq = f_min + (f_max - f_min) * beta
            V[i] = V[i] + (X[i] - best_x) * freq
            sol = np.clip(X[i] + V[i], lb, ub)

            if rng.random() > r[i]:
                eps = rng.uniform(-1.0, 1.0, size=dim)
                sol = np.clip(best_x + eps * mean_A, lb, ub)

            f_new = counter.eval(func, sol)

            if (f_new <= fitness[i]) and (rng.random() < A[i]):
                X[i] = sol
                fitness[i] = f_new
                A[i] = alpha * A[i]
                r[i] = r0 * (1.0 - np.exp(-gamma * t))

            if f_new < best_f:
                best_f = f_new
                best_x = sol.copy()

    elapsed = time.perf_counter() - t_start
    return {
        "best_x": best_x,
        "best_f": best_f,
        "evals": counter.count,
        "hist_evals": np.array(counter.history_evals),
        "hist_best": np.array(counter.history_best),
        "elapsed_s": elapsed,
    }


def random_search(func, lb: np.ndarray, ub: np.ndarray, max_evals: int = 3000, seed: int = 1):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    counter = EvalCounter(max_evals=max_evals)
    t_start = time.perf_counter()

    while counter.remaining > 0:
        x = rng.uniform(lb, ub)
        counter.eval(func, x)

    return {
        "best_f": counter.best_f,
        "hist_evals": np.array(counter.history_evals),
        "hist_best": np.array(counter.history_best),
        "elapsed_s": time.perf_counter() - t_start,
    }