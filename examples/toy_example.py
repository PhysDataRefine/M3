from __future__ import annotations

import numpy as np

from m3 import run_m3


def main() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    xyz = rng.normal(size=(n, 3))
    dense = rng.normal(loc=(0.0, 0.0, 0.0), scale=(0.15, 0.4, 0.4), size=(n // 2, 3))
    xyz[: n // 2] = dense
    phi = xyz[:, 0] + 0.05 * rng.normal(size=n)
    psi = np.column_stack([phi, 0.2 * xyz[:, 1], 0.1 * xyz[:, 2]])

    out = run_m3(
        points_xyz=xyz,
        phi_scalar=phi,
        psi_vec3=psi,
        budget=2048,
        file_type="boundary",
        seed=42,
        rho_fill_ratio=0.3,
        normalize=False,
        min_points_stop_split=3,
    )
    print("n_points:", out.stats["n_points"])
    print("n_cells:", out.n_cells)
    print("n_sampled:", out.n_sampled)
    print("allocation:", out.stats["allocation"])
    print("eff_level unique:", np.unique(out.effective_level))


if __name__ == "__main__":
    main()
