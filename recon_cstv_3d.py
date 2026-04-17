import argparse
import os
import os.path as osp
import time
from typing import Callable, Dict, List, Tuple

import numpy as np


def fft3c(x: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Centered orthonormal 3D FFT."""
    y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm="ortho"))
    return y.astype(out_dtype, copy=False)


def ifft3c(kspace: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Centered orthonormal 3D IFFT."""
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace), norm="ortho"))
    return x.astype(out_dtype, copy=False)


def apply_a(x: np.ndarray, mask: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    return mask * fft3c(x, out_dtype=out_dtype)


def apply_ah(y: np.ndarray, mask: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    return ifft3c(mask * y, out_dtype=out_dtype)


def apply_aha(x: np.ndarray, mask: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    return ifft3c(mask * fft3c(x, out_dtype=out_dtype), out_dtype=out_dtype)


def grad3d(x: np.ndarray) -> np.ndarray:
    """Forward periodic differences along three axes."""
    gx = np.roll(x, shift=-1, axis=0) - x
    gy = np.roll(x, shift=-1, axis=1) - x
    gz = np.roll(x, shift=-1, axis=2) - x
    return np.stack((gx, gy, gz), axis=0)


def div3d(p: np.ndarray) -> np.ndarray:
    """Divergence matching grad3d so that <grad x, p> = <x, -div p>."""
    if p.ndim != 4 or p.shape[0] != 3:
        raise ValueError(f"Expected p shape (3, nx, ny, nz), got {p.shape}")
    dx = p[0] - np.roll(p[0], shift=1, axis=0)
    dy = p[1] - np.roll(p[1], shift=1, axis=1)
    dz = p[2] - np.roll(p[2], shift=1, axis=2)
    return dx + dy + dz


def l2_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.real(np.vdot(x, x))))


def shrink_isotropic(v: np.ndarray, tau: float) -> np.ndarray:
    """Voxel-wise isotropic soft-thresholding for vector field v with axis 0 = components."""
    mag = np.sqrt(np.sum(np.abs(v) ** 2, axis=0))
    denom = np.maximum(mag, 1e-12)
    scale = np.maximum(0.0, 1.0 - (tau / denom))
    return v * scale[None, ...]


def conjugate_gradient(
    linear_op: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int, float]:
    """CG for Hermitian positive semi-definite linear systems."""
    x = x0.copy()
    r = b - linear_op(x)
    p = r.copy()

    rs_old = float(np.real(np.vdot(r, r)))
    b_norm = l2_norm(b) + 1e-12
    rel_res = np.sqrt(rs_old) / b_norm
    if rel_res <= tol:
        return x, 0, rel_res

    for i in range(1, max_iter + 1):
        ap = linear_op(p)
        denom = float(np.real(np.vdot(p, ap)))
        if abs(denom) < 1e-30:
            return x, i - 1, rel_res

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap

        rs_new = float(np.real(np.vdot(r, r)))
        rel_res = np.sqrt(rs_new) / b_norm
        if rel_res <= tol:
            return x, i, rel_res

        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, rel_res


def random_complex(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if dtype == np.complex64:
        real_dtype = np.float32
    else:
        real_dtype = np.float64
    real = rng.standard_normal(shape).astype(real_dtype)
    imag = rng.standard_normal(shape).astype(real_dtype)
    return (real + 1j * imag).astype(dtype)


def run_self_check(seed: int = 0, dtype: np.dtype = np.complex64) -> bool:
    """Sanity checks for adjoint consistency of A/A^H and grad/div."""
    rng = np.random.default_rng(seed)
    shape = (24, 20, 12)

    x = random_complex(rng, shape, dtype)
    y = random_complex(rng, shape, dtype)
    p = random_complex(rng, (3,) + shape, dtype)
    mask = (rng.random(shape) > 0.7).astype(np.float32)

    lhs_a = np.vdot(apply_a(x, mask, out_dtype=dtype), y)
    rhs_a = np.vdot(x, apply_ah(y, mask, out_dtype=dtype))
    adjoint_err = float(abs(lhs_a - rhs_a) / max(abs(lhs_a), abs(rhs_a), 1e-12))

    lhs_g = np.vdot(grad3d(x), p)
    rhs_g = np.vdot(x, -div3d(p))
    grad_div_err = float(abs(lhs_g - rhs_g) / max(abs(lhs_g), abs(rhs_g), 1e-12))

    print(f"[self-check] A/AH adjoint relative error: {adjoint_err:.3e}")
    print(f"[self-check] grad/div relative error:  {grad_div_err:.3e}")

    passed = adjoint_err < 1e-4 and grad_div_err < 1e-4
    print(f"[self-check] status: {'PASS' if passed else 'FAIL'}")
    return passed


def validate_inputs(kspace: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if kspace.ndim != 3:
        raise ValueError(f"kspace must be 3D, got shape {kspace.shape}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")
    if kspace.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: kspace {kspace.shape} vs mask {mask.shape}"
        )
    if not np.iscomplexobj(kspace):
        raise ValueError("kspace must be a complex array (complex64 or complex128)")
    if np.iscomplexobj(mask):
        raise ValueError("mask must be real-valued (0/1)")

    if not np.isfinite(np.real(kspace)).all() or not np.isfinite(np.imag(kspace)).all():
        raise ValueError("kspace contains NaN/Inf values")
    if not np.isfinite(mask).all():
        raise ValueError("mask contains NaN/Inf values")

    non_binary = np.any((mask != 0) & (mask != 1))
    if non_binary:
        print("[warning] mask has non-binary values; thresholding with mask > 0")

    mask_bin = (mask > 0).astype(np.float32)
    sampled = int(mask_bin.sum())
    total = mask_bin.size
    if sampled == 0:
        raise ValueError("mask has no sampled points (all zeros)")

    print(f"[info] sampled points: {sampled}/{total} ({sampled / total:.2%})")

    return kspace, mask_bin


def cs_tv_reconstruct(
    kspace: np.ndarray,
    mask: np.ndarray,
    lambda_tv: float,
    rho: float,
    admm_iters: int,
    cg_iters: int,
    tol: float,
    cg_tol: float,
    verbose_every: int,
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, float]]:
    if lambda_tv < 0:
        raise ValueError("lambda_tv must be non-negative")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if admm_iters <= 0:
        raise ValueError("admm_iters must be positive")
    if cg_iters <= 0:
        raise ValueError("cg_iters must be positive")
    if tol <= 0 or cg_tol <= 0:
        raise ValueError("tol and cg_tol must be positive")

    work_dtype = np.complex64 if kspace.dtype == np.complex64 else np.complex128
    y = (kspace * mask).astype(work_dtype, copy=False)
    mask = mask.astype(np.float32, copy=False)

    ahy = apply_ah(y, mask, out_dtype=work_dtype)
    x = ahy.copy()
    d = grad3d(x)
    u = np.zeros_like(d)

    def normal_op(z: np.ndarray) -> np.ndarray:
        return apply_aha(z, mask, out_dtype=work_dtype) - rho * div3d(grad3d(z))

    history: List[Dict[str, float]] = []
    start = time.perf_counter()

    for it in range(1, admm_iters + 1):
        rhs = ahy - rho * div3d(d - u)
        x, cg_used, cg_rel = conjugate_gradient(
            normal_op,
            rhs,
            x0=x,
            max_iter=cg_iters,
            tol=cg_tol,
        )

        grad_x = grad3d(x)
        d_prev = d
        d = shrink_isotropic(grad_x + u, tau=lambda_tv / rho)
        u = u + grad_x - d

        primal = l2_norm(grad_x - d)
        dual = rho * l2_norm(-div3d(d - d_prev))

        primal_rel = primal / (l2_norm(grad_x) + 1e-12)
        dual_rel = dual / (l2_norm(-div3d(u)) + 1e-12)

        history.append(
            {
                "iter": float(it),
                "primal": primal,
                "dual": dual,
                "primal_rel": primal_rel,
                "dual_rel": dual_rel,
                "cg_iters": float(cg_used),
                "cg_rel": cg_rel,
                "elapsed_sec": time.perf_counter() - start,
            }
        )

        should_log = (
            verbose_every > 0
            and (it == 1 or it % verbose_every == 0 or it == admm_iters)
        )
        if should_log:
            print(
                "[iter {:4d}] primal={:.3e} dual={:.3e} "
                "primal_rel={:.3e} dual_rel={:.3e} cg_iters={} cg_rel={:.3e}".format(
                    it,
                    primal,
                    dual,
                    primal_rel,
                    dual_rel,
                    int(cg_used),
                    cg_rel,
                )
            )

        if max(primal_rel, dual_rel) < tol:
            print(f"[info] ADMM converged at iter {it} with tol={tol:.1e}")
            break

    elapsed = time.perf_counter() - start
    last = history[-1]
    info = {
        "iters": float(len(history)),
        "elapsed_sec": elapsed,
        "final_primal": float(last["primal"]),
        "final_dual": float(last["dual"]),
        "final_primal_rel": float(last["primal_rel"]),
        "final_dual_rel": float(last["dual_rel"]),
    }
    return x, history, info


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CS-TV reconstruction for undersampled 3D MRI k-space"
    )
    parser.add_argument(
        "--kspace",
        type=str,
        default="MRIdata/acc_rate8_sigma30/kspace_gt.npy",
        help="Path to undersampled complex k-space .npy",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="MRIdata/acc_rate8_sigma30/mask_3D.npy",
        help="Path to 3D sampling mask .npy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="MRIdata/acc_rate8_sigma30/recon_cstv_3d_complex.npy",
        help="Output path for complex reconstruction (.npy)",
    )
    parser.add_argument("--lambda-tv", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=1e-2)
    parser.add_argument("--admm-iters", "-i", type=int, default=100)
    parser.add_argument("--cg-iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--verbose-every", type=int, default=10)

    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run numerical adjoint checks before reconstruction",
    )
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="Run only numerical checks and exit",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.self_check or args.self_check_only:
        ok = run_self_check(seed=args.seed, dtype=np.complex64)
        if not ok:
            raise SystemExit(2)
        if args.self_check_only:
            return

    if not os.path.exists(args.kspace):
        raise FileNotFoundError(f"kspace file not found: {args.kspace}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"mask file not found: {args.mask}")

    print(f"[info] loading kspace: {args.kspace}")
    kspace = np.load(args.kspace)
    print(f"[info] loading mask:   {args.mask}")
    mask = np.load(args.mask)

    kspace, mask = validate_inputs(kspace, mask)
    print(f"[info] kspace dtype={kspace.dtype}, shape={kspace.shape}")

    start = time.perf_counter()
    recon, history, info = cs_tv_reconstruct(
        kspace=kspace,
        mask=mask,
        lambda_tv=args.lambda_tv,
        rho=args.rho,
        admm_iters=args.admm_iters,
        cg_iters=args.cg_iters,
        tol=args.tol,
        cg_tol=args.cg_tol,
        verbose_every=args.verbose_every,
    )
    elapsed = time.perf_counter() - start

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    recon_to_save = recon.astype(np.complex64, copy=False)
    np.save(args.out, recon_to_save)
    np.save(osp.join(out_dir, "recon_cstv_3d_magn.npy"), np.abs(recon_to_save))

    print(f"[done] Saved complex reconstruction to: {args.out}")
    print(f"[done] Saved magnitude reconstruction to: {osp.join(out_dir, 'recon_cstv_3d_magn.npy')}")
    print(
        "[done] Total time: {:.2f}s | ADMM iters: {} | final primal_rel={:.3e}, dual_rel={:.3e}".format(
            elapsed,
            int(info["iters"]),
            info["final_primal_rel"],
            info["final_dual_rel"],
        )
    )
    if len(history) == 0:
        print("[warning] No ADMM history recorded")


if __name__ == "__main__":
    main()
