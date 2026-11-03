"""Production-only LOC1 helpers extracted from hv2f4 rg3f_lc; no truth RPCs."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from . import combined_tools as CT, align_target
FIX_VIEWS = (("A", 0.13, -15.0), ("B", 0.17, 20.0))
OPEN_MIN, CLOSED_MAX, CLEAR_Z = 0.9, 0.02, 0.03
SPEED_AT, SETTLE_AT, UNLOAD_AT = 0.002, 6, 3
DET_AT = {}  # R1 uses its frozen stall rule; R0 requires an explicit detector profile.
def _at(): return align_target

def _coarse(co, pk):
    def call(paths):
        req = dict(images_b64={c: CT._b64(p) for c, p in paths.items()}, states=pk['current_state'],
                   eef_positions=[pk['current_eef'][a]['position'] for a in CT.ARMS], eef_quaternions_wxyz=[pk['current_eef'][a]['quaternion_wxyz'] for a in CT.ARMS],
                   object='power strip', camera='cam_head')
        return CT._post(co.cfg['locate_url'], req, timeout=120)
    loc = call(CT._imgs(pk)); src = 'current'
    if (not loc.get('found') or loc.get('center_xyz') is None) and co.start_imgs:
        loc = call(co.start_imgs); src = 'episode start images'
    return loc, src


def _joints(ad, ar):
    i = CT.ARMS.index(ar); return np.asarray(ad.packet['current_state'][7 * i: 7 * i + 6], float)


def _q_of(M):
    x, y, z, w = R.from_matrix(np.asarray(M)[:3, :3]).as_quat(); return [float(w), float(x), float(y), float(z)]
