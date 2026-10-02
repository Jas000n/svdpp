"""SVD++ (Koren, KDD 2008) trained by SGD on MovieLens-100K, evaluated with 5-fold CV.

    python svdpp.py                                   # SVD++  -> results/svdpp.json
    python svdpp.py --k 0 --out results/baseline.json # biases only (mu + b_u + b_i)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ml-100k"


def read_info(data_dir):
    """Number of users and items, from u.info (ids in every split are 1..n)."""
    counts = {}
    for line in (data_dir / "u.info").read_text().splitlines():
        n, name = line.split()
        counts[name] = int(n)
    return counts["users"], counts["items"]


def load_ratings(path):
    """0-based user ids, 0-based item ids and ratings of a u.data-format file."""
    a = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2))
    return a[:, 0] - 1, a[:, 1] - 1, a[:, 2].astype(np.float64)


@njit(cache=True)
def sgd_epoch(user_order, ptr, items, ratings, within, mu, bu, bi, p, q, y, lr, l1, l2):
    """One SGD pass over the training ratings, grouped by user.

    The ratings of user u are items[ptr[u]:ptr[u+1]] (visited in the order given by
    `within`), which is also R(u). Every step updates y_j for all j in R(u) with
        y_j <- (1 - lr*l2) * y_j + lr * g_t,
    so instead of touching |R(u)| rows per step we keep the running sum
    s = sum_{j in R(u)} y_j up to date in O(k) and apply the accumulated update
    y_j <- d * y_j + acc to each y_j once, after the user's last rating. This is
    exactly per-rating SGD, at O(k) per rating instead of O(|R(u)| * k).
    """
    k = p.shape[1]
    keep = 1.0 - lr * l2
    s = np.empty(k)
    acc = np.empty(k)
    pz = np.empty(k)
    for u in user_order:
        lo, hi = ptr[u], ptr[u + 1]
        n = hi - lo
        if n == 0:
            continue
        norm = 1.0 / np.sqrt(n)
        s[:] = 0.0
        for t in range(lo, hi):
            s += y[items[t]]
        acc[:] = 0.0
        d = 1.0
        for t in within[lo:hi]:
            i = items[t]
            dot = 0.0
            for f in range(k):
                pz[f] = p[u, f] + norm * s[f]
                dot += q[i, f] * pz[f]
            e = ratings[t] - (mu + bu[u] + bi[i] + dot)
            bu[u] += lr * (e - l1 * bu[u])
            bi[i] += lr * (e - l1 * bi[i])
            for f in range(k):
                qf = q[i, f]  # p_u and y_j use q_i from before this step
                q[i, f] += lr * (e * pz[f] - l2 * qf)
                p[u, f] += lr * (e * qf - l2 * p[u, f])
                g = lr * e * norm * qf
                acc[f] = keep * acc[f] + g
                s[f] = keep * s[f] + n * g
            d *= keep
        for t in range(lo, hi):
            j = items[t]
            for f in range(k):
                y[j, f] = d * y[j, f] + acc[f]


def run_fold(train, test, n_users, n_items, k, epochs, lr, decay, l1, l2, init_std, rng):
    """Train on `train`, return test RMSE/MAE after init and after every epoch."""
    users, items, ratings = train
    order = np.argsort(users, kind="stable")
    users, items, ratings = users[order], items[order], ratings[order]
    ptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(users, minlength=n_users), out=ptr[1:])

    # |R(u)|^-1/2 * sum_{j in R(u)} y_j for every user at once is implicit @ y
    implicit = np.zeros((n_users, n_items))
    implicit[users, items] = np.diff(ptr)[users] ** -0.5

    mu = ratings.mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    p = rng.normal(0.0, init_std, (n_users, k))
    q = rng.normal(0.0, init_std, (n_items, k))
    y = rng.normal(0.0, init_std, (n_items, k))

    test_users, test_items, test_ratings = test
    history = {"rmse": [], "mae": [], "train_time": [], "test_time": []}

    def evaluate():
        t0 = time.perf_counter()
        z = p[test_users] + (implicit @ y)[test_users]
        pred = mu + bu[test_users] + bi[test_items] + np.einsum("ij,ij->i", q[test_items], z)
        err = test_ratings - pred
        history["rmse"].append(float(np.sqrt(np.mean(err ** 2))))
        history["mae"].append(float(np.mean(np.abs(err))))
        history["test_time"].append(time.perf_counter() - t0)

    # compile outside the timed region
    sgd_epoch(np.empty(0, np.int64), ptr, items, ratings, np.arange(len(users)),
              mu, bu, bi, p, q, y, lr, l1, l2)
    evaluate()
    for _ in range(epochs):
        user_order = rng.permutation(n_users)
        within = np.lexsort((rng.random(len(users)), users))
        t0 = time.perf_counter()
        sgd_epoch(user_order, ptr, items, ratings, within, mu, bu, bi, p, q, y, lr, l1, l2)
        history["train_time"].append(time.perf_counter() - t0)
        evaluate()
        lr *= decay
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=50, help="latent dimension; 0 gives the bias-only baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.007)
    parser.add_argument("--decay", type=float, default=0.9, help="learning rate decay per epoch")
    parser.add_argument("--l1", type=float, default=0.005, help="regularization of b_u, b_i")
    parser.add_argument("--l2", type=float, default=0.015, help="regularization of p_u, q_i, y_j")
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "svdpp.json")
    args = parser.parse_args()

    n_users, n_items = read_info(DATA_DIR)
    rng = np.random.default_rng(args.seed)
    folds = []
    for fold in args.folds:
        train = load_ratings(DATA_DIR / f"u{fold}.base")
        test = load_ratings(DATA_DIR / f"u{fold}.test")
        history = run_fold(train, test, n_users, n_items, args.k, args.epochs, args.lr, args.decay,
                           args.l1, args.l2, args.init_std, rng)
        folds.append({"fold": fold, **history})
        print(f"fold {fold}: rmse={history['rmse'][-1]:.4f}  mae={history['mae'][-1]:.4f}  "
              f"train={sum(history['train_time']):.2f}s  test={sum(history['test_time']):.3f}s")

    rmse = np.mean([f["rmse"][-1] for f in folds])
    mae = np.mean([f["mae"][-1] for f in folds])
    train_time = sum(sum(f["train_time"]) for f in folds)
    test_time = sum(sum(f["test_time"]) for f in folds)
    print(f"mean : rmse={rmse:.4f}  mae={mae:.4f}  train={train_time:.2f}s  test={test_time:.3f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()}
    args.out.write_text(json.dumps({"config": config, "folds": folds}, indent=1))


if __name__ == "__main__":
    main()
