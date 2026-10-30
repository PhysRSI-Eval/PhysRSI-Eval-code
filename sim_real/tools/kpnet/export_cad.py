"""keypoint-tool factory copy of finepose src/export_cad.py (sha 78a74c714c054906): assets given as NAME=REL args.
  python export_cad.py OUT charger=Rigid/charger/00000 socket=Geometry/socket/00000"""
"""finepose (10-29, user: use the key / lock CAD directly): export the visible meshes (vertices + triangulated faces, object frame, metres) of the
key (Rigid/key/00000) and the lock (Geometry/key_slot/00000) from the RoboDojo USD assets to OBJ + NPZ. Read-only on the assets."""
import json, sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
A = Path('/opt/PhysRSI/runtime/RoboDojo/Assets/Object/RoboDojo'); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name, rel in [x.split('=') for x in sys.argv[2:]]:
    st = Usd.Stage.Open(str(A / rel / 'object.usdz')); mpu = UsdGeom.GetStageMetersPerUnit(st); cache = UsdGeom.XformCache(Usd.TimeCode.Default()); V, Fs, parts = [], [], []
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible: continue
        m = UsdGeom.Mesh(prim); pts = np.array(m.GetPointsAttr().Get(), float); M = np.array(cache.GetLocalToWorldTransform(prim), float)
        P = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu; cnt = np.array(m.GetFaceVertexCountsAttr().Get()); idx = np.array(m.GetFaceVertexIndicesAttr().Get()); o = sum(len(v) for v in V); k = 0
        for c in cnt:  # fan-triangulate polygons
            f = idx[k:k + c]; Fs += [[o + f[0], o + f[i], o + f[i + 1]] for i in range(1, c - 1)]; k += c
        V.append(P); parts.append(dict(prim=str(prim.GetPath()), n_vertices=len(P)))
    V = np.vstack(V); F = np.array(Fs, int)
    np.savez(out / f'{name}.npz', vertices=V, faces=F)
    with open(out / f'{name}.obj', 'w') as f:
        f.write(f'# {name} visible mesh from {rel} (object frame, metres), exported 2026-10-29\n'); f.writelines('v %.6f %.6f %.6f\n' % tuple(v) for v in V); f.writelines('f %d %d %d\n' % tuple(t + 1) for t in F)
    info = dict(asset=rel, n_vertices=len(V), n_faces=len(F), bbox_min=V.min(0).round(5).tolist(), bbox_max=V.max(0).round(5).tolist(), meters_per_unit=mpu, parts=parts)
    (out / f'{name}.json').write_text(json.dumps(info, indent=1)); print(name, json.dumps({k: info[k] for k in ('n_vertices', 'n_faces', 'bbox_min', 'bbox_max')}))
