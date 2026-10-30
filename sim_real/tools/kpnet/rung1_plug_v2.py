"""KPF-plug-v2 rung 1 (PREREG-KPF-plug-v2). Per state: ALL captured views (free-arm parks A/B when reached, holding wrist, head), each
with its view-type keypoint model (22 points: kpnet3 or the DINOv3 frozen-feature head dinokp; --pred truth = the labels' visible pixels,
the ceiling). Visible (p >= 0.5) points are triangulated across views (>= 2 views); the prong tips are recovered by a rigid CAD fit
(Kabsch) of all triangulated charger points (tips, roots, face, 8 housing corners; >= 3 points) -> tips = T * tips_CAD; the target
outlet = the outlet the executor aims at (scored as the truth outlet nearest the tips); relative error = mean (hole entry - tip).
Views are restricted to the largest static subset of the state (charger and strip truth within 1 mm). Scored against the truth keypoints of that subset. Refuses to write a TEST verdict twice or on a non-judgeable TEST.
  python rung1_plug_v2.py --pred kpnet|dino|truth --split test|dev --out FILE --gpu G"""
import argparse, glob, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
K = Path('/lizekai/RSIws/experiments/kpfactory-20261029'); sys.path.insert(0, str(K / 'src'))
import kp_api as A  # noqa: E402
CH = list(range(5)) + list(range(14, 22)); TIPS_CAD = np.array([[-0.0055, -0.0152, 0.0], [0.0055, -0.0152, 0.0]])
CAD = np.vstack([TIPS_CAD, [[-0.0055, 0, 0], [0.0055, 0, 0], [0, 0, 0]], [[x, y, z] for y in (0.0, 0.0479) for x in (-0.021, 0.021) for z in (-0.0135, 0.0135)]])
GROUP = lambda r: 'head' if r['camera'] == 'cam_high' else ('hold' if '-hold-' in r['sample_id'] else 'free')


def kabsch(Pm, Pw):
    cm, cw = Pm.mean(0), Pw.mean(0); U, S, Vt = np.linalg.svd((Pm - cm).T @ (Pw - cw)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    return R, cw - R @ cm


def solve(P, target=None):
    idx = [k for k in CH if not np.isnan(P[k]).any()]; ci = [CH.index(k) for k in idx]
    if len(idx) < 3 or np.linalg.matrix_rank(P[idx] - P[idx].mean(0), tol=1e-3) < 2: return None
    R, t = kabsch(CAD[ci], P[idx]); tips = TIPS_CAD @ R.T + t; best = None
    for o in (range(3) if target is None else [target]):  # the executor knows which outlet it is aiming at
        H = P[[5 + 3 * o, 6 + 3 * o]]
        if np.isnan(H).any(): continue
        pr = min(((0, 1), (1, 0)), key=lambda q: np.linalg.norm(tips[0, :2] - H[q[0], :2]) + np.linalg.norm(tips[1, :2] - H[q[1], :2])); Hm = H[list(pr)]
        d = float(np.linalg.norm((Hm[:, :2] - tips[:, :2]).mean(0)))
        if best is None or d < best[0]: best = (d, o, (Hm[:, :2] - tips[:, :2]).mean(0), float((tips[:, 2] - Hm[:, 2]).mean()))
    if best is None: return None
    return dict(outlet=best[1], dxy=best[2], dz=best[3], tips=tips, n_charger_pts=len(idx))


def main(a):
    out = Path(a.out); assert not (a.split == 'test' and out.exists()), 'already judged'
    sub = 'test' if a.split == 'test' else 'trdev'; recs = []
    for g in ('free', 'hold', 'head'):
        recs += [json.loads(l) for f in sorted(glob.glob(str(K / 'kp-charger-v2' / g / sub / 'L*.jsonl'))) for l in open(f)]
    if a.split == 'dev': recs = [r for r in recs if r['split'] == 1]
    ck = {g: str(K / 'models' / ('kp2-%s-%s.pt' % (a.pred, g))) for g in ('free', 'hold', 'head')}; dets = {}
    if a.pred == 'dino':
        import dinokp
        dets = {g: dinokp.Detector(ck[g], a.gpu) for g in ck}
    states = {}
    for r in recs: states.setdefault((r['episode'], r['state']), []).append(r)
    rows = []; px = {g: [] for g in ('free', 'hold', 'head')}
    for key, rs in sorted(states.items()):
        # the free arm's parks can bump the strip / the swinging charger: keep the largest subset of views whose charger AND strip
        # truth positions agree within 1 mm (a static scene), as triangulation requires
        pos = [np.r_[r['outlets'][0], r['outlets'][7]] for r in rs]; best = []
        for i in range(len(rs)):
            grp = [j for j in range(len(rs)) if np.abs(np.asarray(pos[j]) - np.asarray(pos[i])).max() < 0.001]
            if len(grp) > len(best): best = grp
        n_all = len(rs); rs = [rs[j] for j in best]
        D = []; cams = []
        for r in rs:
            g = GROUP(r); uvt = np.asarray([u if u is not None else [np.nan, np.nan] for u in r['outlets_uv']], float)
            if a.pred == 'truth': d = [(uvt[k, 0], uvt[k, 1], 1.0 if r['in_image'][k] else 0.0, 1.0) for k in range(22)]
            else:
                im = np.asarray(Image.open(K / 'data-charger-hover' / r['image']).convert('RGB'))
                d = dets[g](im) if a.pred == 'dino' else A.detect(ck[g], [im], a.gpu)[0]
                for k in range(22):
                    if r['in_image'][k]: px[g].append(float(np.hypot(d[k][0] - uvt[k, 0], d[k][1] - uvt[k, 1])))
            D.append(d); cams.append((r['intrinsics'], r['extrinsics_cam_to_world']))
        Pt = np.asarray(rs[0]['outlets'], float); row = dict(episode=key[0], state=key[1], kind=rs[0]['kind'], views=[GROUP(r) for r in rs], n_views_all=n_all)
        P, res = A.triangulate(D, cams) if len(D) >= 2 else (np.full((22, 3), np.nan), None); Tt0 = Pt[[0, 1]]; tgt = int(np.argmin([np.linalg.norm(Pt[7 + 3 * k][:2] - Tt0.mean(0)[:2]) for k in range(3)])); s = solve(P, tgt); row['covered'] = s is not None
        if s is not None:
            o = s['outlet']; Ht = Pt[[5 + 3 * o, 6 + 3 * o]]; Tt = Pt[[0, 1]]; pr = min(((0, 1), (1, 0)), key=lambda q: np.linalg.norm(Tt[0, :2] - Ht[q[0], :2]) + np.linalg.norm(Tt[1, :2] - Ht[q[1], :2]))
            dxy_t = (Ht[list(pr)][:, :2] - Tt[:, :2]).mean(0); dz_t = float((Tt[:, 2] - Ht[list(pr)][:, 2]).mean())
            row.update(outlet=o, dxy_err_mm=float(np.linalg.norm(s['dxy'] - dxy_t)) * 1000, dz_err_mm=abs(s['dz'] - dz_t) * 1000,
                       tip_err_mm=float(np.linalg.norm(s['tips'] - Tt, axis=1).mean()) * 1000, n_charger_pts=s['n_charger_pts'],
                       outlet_true=int(np.argmin([np.linalg.norm(Pt[7 + 3 * k][:2] - Tt.mean(0)[:2]) for k in range(3)])))
        rows.append(row)
    cov = [r for r in rows if r['covered']]; S = dict(pred=a.pred, split=a.split, states=len(rows), episodes=len({r['episode'] for r in rows}), coverage=round(len(cov) / max(1, len(rows)), 3))
    allpx = sum(px.values(), [])
    if allpx: S.update(px_median=round(float(np.median(allpx)), 3), px_p90=round(float(np.percentile(allpx, 90)), 3), px_by_view={g: round(float(np.median(v)), 3) for g, v in px.items() if v})
    for k in ('dxy_err_mm', 'dz_err_mm', 'tip_err_mm'):
        v = [r[k] for r in cov]
        if v: S[k] = dict(median=round(float(np.median(v)), 3), p90=round(float(np.percentile(v, 90)), 3))
    if cov: S['outlet_correct'] = round(float(np.mean([r['outlet'] == r['outlet_true'] for r in cov])), 3)
    S['judgeable'] = bool(len(rows) >= 20 and S['episodes'] >= 3)
    if a.split == 'test' and not S['judgeable']: print('NOT JUDGEABLE', json.dumps(S)); sys.exit(3)
    if a.split == 'test':
        S['PASS'] = bool(S.get('px_median', 99) <= 1.5 and S['coverage'] >= 0.5 and S.get('dxy_err_mm', {}).get('median', 99) <= 1.0 and S.get('dxy_err_mm', {}).get('p90', 99) <= 2.5)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(dict(prereg='PREREG-KPF-plug-v2.md', summary=S, rows=rows), indent=1)); print(json.dumps(S))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--split', default='dev'); ap.add_argument('--out', required=True); ap.add_argument('--gpu', type=int, default=0)
    main(ap.parse_args())
