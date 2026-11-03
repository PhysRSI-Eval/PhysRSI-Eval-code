"""combined-online-20261026 inference service (deploy-time, RGB + proprio only; torch env with sm_120).
Endpoints (JSON over HTTP POST, images as base64 PNG/JPEG):
  GET  /health   -> {ok, outlets: loaded?, inhand: loaded?, versions}
  POST /outlets  {image_b64}                       -> {detections: [[u, v, visible_prob, peak] x 3]}  (LOC1 frozen keypoint net, outlet order fixed)
  POST /inhand   {images_b64: {cam_high, cam_<arm>_wrist}, states[14], eef_positions[2][3], eef_quaternions_wxyz[2][4], arm, category}
                 -> {position_m[3], quaternion_wxyz[4] (object in the TCP frame), position_sigma_m, rotation_sigma_deg, model}
Model files are given by environment variables and their sha256 is reported in /health:
  CO_KP_CKPT   LOC1 kpnet checkpoint (experiments/loc1-20261026/src/kpnet.py Detector)
  CO_POSE_PKG  in-hand pose v1 package directory (experiments/inhand-pose-20261026/model_v1/pkg, infer_v1.load/infer)
python co_infer_service.py --port P --gpu G"""
import argparse, base64, hashlib, io, json, os, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
LOCK = threading.Lock(); M = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def img(b):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB'))


def load(gpu):
    kp = os.environ.get('CO_KP_CKPT')
    if kp and Path(kp).is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parent)); import kpnet  # v1.3: byte-identical frozen copy of loc1-20261026/src/kpnet.py (svc/kpnet.py)
        M['kp'] = kpnet.Detector(kp, gpu); M['kp_sha'] = sha(kp)
    pd = os.environ.get('CO_POSE_PKG')
    if pd and (Path(pd) / 'infer_v1.py').is_file():
        sys.path.insert(0, pd); import infer_v1
        torch_dev = 'cuda:%d' % gpu
        M['pose'] = (infer_v1, infer_v1.load(device=torch_dev)); M['pose_ver'] = 'inhand-pose v1 (FROZEN 2026-10-26T15:03:52Z, sha256-verified 3 members)'


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/health'):
            return self._send(200, dict(ok=True, outlets='kp' in M, inhand='pose' in M, kp_sha256=M.get('kp_sha'), pose_version=M.get('pose_ver')))
        self._send(404, dict(error='not found'))

    def do_POST(self):
        t = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.startswith('/outlets'):
                if 'kp' not in M: return self._send(503, dict(error='outlet keypoint model not loaded'))
                with LOCK: d = M['kp'](img(req['image_b64']))
                return self._send(200, dict(detections=[list(x) for x in d], seconds=round(time.time() - t, 3)))
            if self.path.startswith('/inhand'):
                if 'pose' not in M: return self._send(503, dict(error='in-hand pose model not loaded'))
                import tempfile
                mod, est = M['pose']; arm = req['arm']; i = ('left', 'right').index(arm)
                with tempfile.TemporaryDirectory() as td:
                    paths = {}
                    for k, v in req['images_b64'].items():
                        paths[k] = str(Path(td) / (k + '.png')); Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB').save(paths[k])
                    with LOCK:
                        o = mod.infer(est, wrist=paths['cam_%s_wrist' % arm], high=paths['cam_high'], link6_pos=req['eef_positions'][i],
                                      link6_quat_wxyz=req['eef_quaternions_wxyz'][i], gripper_state=req['states'][7 * i + 6],
                                      gripper_cmd=float(req.get('gripper_cmd', 0.0)), arm=arm, category=req.get('category', 'charger'))
                o = json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
                std = o.get('pos_std_mm'); std = float(np.linalg.norm(std)) if isinstance(std, list) else std
                return self._send(200, dict(position_m=[v / 1000.0 for v in o['pos_mm']], quaternion_wxyz=o['quat_wxyz'],
                                            position_sigma_m=None if std is None else std / 1000.0, rotation_sigma_deg=o.get('rot_std_deg'),
                                            member_spread_mm=o.get('member_spread_mm'), member_spread_deg=o.get('member_spread_deg'), raw=o,
                                            seconds=round(time.time() - t, 3)))
            self._send(404, dict(error='not found'))
        except Exception as e:  # noqa: BLE001
            self._send(500, dict(error='%s: %s' % (type(e).__name__, str(e)[:300]), trace=traceback.format_exc()[-800:]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, required=True); ap.add_argument('--gpu', type=int, default=0); a = ap.parse_args()
    load(a.gpu); print('loaded', {k: v for k, v in M.items() if isinstance(v, str)}, flush=True)
    ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()
