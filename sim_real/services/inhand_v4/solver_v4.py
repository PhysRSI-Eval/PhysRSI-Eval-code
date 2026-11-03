"""in-hand pose v4 multi-view pose solver (batched, torch). Pose G = object frame in link6. Residuals: predicted keypoint pixels in the wrist
image (320x240) and the head crop (160x160) vs. the projection of the CAD keypoints, weighted by the predicted scale b (Laplace), Huber IRLS.
Levenberg-Marquardt on se(3) with a numeric Jacobian (7 residual evaluations per iteration). Flip check: the chosen solution's three 180-deg
flips about the CAD principal axes (through the CAD centroid) are refined too; the flip margin = min cost of a refined alternative that is
still > 90 deg away (directed axis for axial objects) / best cost."""
import math
import numpy as np
import torch

HUBER = 2.5


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9); k = w / th; K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K); s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


class Geo:
    """per-sample projection constants (float64 tensors on device)."""
    def __init__(self, L, crop, head_cam, wrist, dev):
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)  # noqa: E731
        self.L = f(L); self.crop = f(crop); Kh = np.asarray(head_cam['intrinsics']); C2W = np.asarray(head_cam['extrinsics_cam_to_world']); W2C = np.linalg.inv(C2W)
        self.W2C = f(W2C); self.Kh = f(Kh); self.R_LC = f(wrist['R_LC']); self.T_LC = f(wrist['T_LC']); self.fw, self.cxw, self.cyw = wrist['fx'], wrist['cx'], wrist['cy']

    def project(self, X, idx=None):
        """X: (B, N, 3) link6-frame points -> wrist uv (B,N,2), head-crop uv (B,N,2), wrist depth."""
        L = self.L if idx is None else self.L[idx]; crop = self.crop if idx is None else self.crop[idx]
        pc = (X - self.T_LC) @ self.R_LC; z = pc[..., 2].clamp_min(1e-3)
        uvw = torch.stack([self.fw * pc[..., 0] / z + self.cxw, self.fw * pc[..., 1] / z + self.cyw], -1)
        Xw = X @ L[:, :3, :3].transpose(1, 2) + L[:, None, :3, 3]; q = Xw @ self.W2C[:3, :3].T + self.W2C[:3, 3]; zh = (-q[..., 2]).clamp_min(1e-3)
        u = self.Kh[0, 0] * q[..., 0] / zh + self.Kh[0, 2]; v = -self.Kh[1, 1] * q[..., 1] / zh + self.Kh[1, 2]
        S = crop[:, None, 2]; uvh = torch.stack([(u - crop[:, None, 0]) * 160 / S, (v - crop[:, None, 1]) * 160 / S], -1)
        return uvw, uvh, pc[..., 2]


def residuals(geo, R, t, kp, uvw, bw, uvh, bh):
    """R (B,3,3), t (B,3), kp (B,K,3). Returns whitened residuals (B, 4K) and raw px residuals (wrist px; head crop px x2)."""
    X = kp @ R.transpose(1, 2) + t[:, None]; pw, ph, zw = geo.project(X)
    rw = (pw - uvw) / bw[..., None]; rh = (ph - uvh) / bh[..., None]
    rw = torch.where((zw > 0.01)[..., None], rw, torch.zeros_like(rw))
    return torch.cat([rw.flatten(1), rh.flatten(1)], 1)


def lm(geo, R, t, kp, uvw, bw, uvh, bh, iters=15):
    lam = torch.full((R.shape[0], 1, 1), 1e-3, dtype=R.dtype, device=R.device); eps = 1e-5
    for _ in range(iters):
        r0 = residuals(geo, R, t, kp, uvw, bw, uvh, bh); a = r0.abs(); wts = torch.where(a > HUBER, HUBER / a.clamp_min(1e-9), torch.ones_like(a)).sqrt()
        J = []
        for j in range(6):
            d = torch.zeros(R.shape[0], 6, dtype=R.dtype, device=R.device); d[:, j] = eps
            Rj = so3_exp(d[:, :3]) @ R; tj = t + d[:, 3:] * 0.1
            J.append((residuals(geo, Rj, tj, kp, uvw, bw, uvh, bh) - r0) / eps)
        J = torch.stack(J, -1) * wts[..., None]; rr = r0 * wts
        H = J.transpose(1, 2) @ J; g = J.transpose(1, 2) @ rr[..., None]
        dx = -torch.linalg.solve(H + lam * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2) + 1e-6), g)[..., 0]
        Rn = so3_exp(dx[:, :3]) @ R; tn = t + dx[:, 3:] * 0.1
        c0 = cost(residuals(geo, R, t, kp, uvw, bw, uvh, bh)); c1 = cost(residuals(geo, Rn, tn, kp, uvw, bw, uvh, bh)); ok = (c1 < c0)
        R = torch.where(ok[:, None, None], Rn, R); t = torch.where(ok[:, None], tn, t); lam = torch.where(ok[:, None, None], lam * 0.3, lam * 10).clamp(1e-6, 1e6)
    return R, t, cost(residuals(geo, R, t, kp, uvw, bw, uvh, bh))


def cost(r):
    a = r.abs(); return torch.where(a > HUBER, 2 * HUBER * a - HUBER ** 2, a ** 2).mean(1)


def flip_mats(axes):
    """axes (B,3,3) rows = principal axes -> (B,3,3,3) rotations by pi about each."""
    return torch.stack([2 * axes[:, i, :, None] * axes[:, i, None, :] - torch.eye(3, dtype=axes.dtype, device=axes.device) for i in range(3)], 1)


def apply_flip(R, t, F, ctr):
    """object-frame flip about the centroid: G' = G * [F, c - F c]."""
    return R @ F, t + (R @ (ctr[..., None] - F @ ctr[..., None]))[..., 0]


def rot_dist(Ra, Rb, axis=None):
    if axis is not None:
        a = (Ra @ axis[..., None])[..., 0]; b = (Rb @ axis[..., None])[..., 0]; return torch.rad2deg(torch.arccos((a * b).sum(-1).clamp(-1, 1)))
    c = ((Ra.transpose(1, 2) @ Rb).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2; return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def solve(geo, kp, ctr, axes, axis, axial, uvw, bw, uvh, bh, R0, t0):
    """full pipeline: 4 starts (regression + 3 flips) -> best; refine the 3 flips of the best -> flip margin. axis (B,3) zeros for non-axial."""
    F = flip_mats(axes); starts = [(R0, t0)] + [apply_flip(R0, t0, F[:, i], ctr) for i in range(3)]
    sols = [lm(geo, R, t, kp, uvw, bw, uvh, bh) for R, t in starts]; C = torch.stack([s[2] for s in sols], 1); best = C.argmin(1)
    B = R0.shape[0]; ar = torch.arange(B, device=R0.device)
    Rb = torch.stack([s[0] for s in sols], 1)[ar, best]; tb = torch.stack([s[1] for s in sols], 1)[ar, best]; cb = C[ar, best]
    alt = torch.full((B,), float('inf'), dtype=R0.dtype, device=R0.device)
    for i in range(3):
        Ra, ta = apply_flip(Rb, tb, F[:, i], ctr); Ra, ta, ca = lm(geo, Ra, ta, kp, uvw, bw, uvh, bh)
        d = torch.where(axial, rot_dist(Ra, Rb, axis), rot_dist(Ra, Rb)); alt = torch.where(d > 90, torch.minimum(alt, ca), alt)
    for s in sols:
        d = torch.where(axial, rot_dist(s[0], Rb, axis), rot_dist(s[0], Rb)); alt = torch.where(d > 90, torch.minimum(alt, s[2]), alt)
    return Rb, tb, cb, alt / cb.clamp_min(1e-9)
