"""In-hand pose v4 (charger only) HTTP service, request-compatible with the :24818 service (co_infer_service_v3.py /inhand):
  POST /inhand {images_b64: {cam_high, cam_<arm>_wrist}, states[14], eef_positions[2][3], eef_quaternions_wxyz[2][4], arm, category}
  -> {position_m (TCP frame), quaternion_wxyz, member_spread_deg, uncertain, abstain, flip_flag, uncertainty, note, seconds}
  GET /health
Run: CUDA_VISIBLE_DEVICES=<g> /opt/PhysRSI/ENV/robodojo/bin/python svc_inhand_v4.py --port P   (port not registered yet; pick with the dispatcher)"""
import argparse, base64, io, json, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import infer_v4 as IV  # noqa: E402
EST = {}; LOCK = threading.Lock()


def img(b): return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB'))


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a): pass

    def do_GET(self):
        self._send(200, dict(ok=True, model='inhand_pose v4 (charger)', frozen=json.load(open(IV.FROZEN))['members'])) if self.path.startswith('/health') else self._send(404, {})

    def do_POST(self):
        t = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if not self.path.startswith('/inhand'): return self._send(404, dict(error='not found'))
            arm = req['arm']; i = ('left', 'right').index(arm)
            with LOCK:
                o = IV.infer(EST['e'], wrist=img(req['images_b64']['cam_%s_wrist' % arm]), high=img(req['images_b64']['cam_high']), link6_pos=req['eef_positions'][i],
                             link6_quat_wxyz=req['eef_quaternions_wxyz'][i], gripper_state=req['states'][7 * i + 6], arm=arm, category=req.get('category', 'charger'))
            if o.get('value') is None and o.get('full') is None: return self._send(200, dict(abstain=True, note=o['note']))
            f = o['full']
            self._send(200, dict(position_m=f['position_in_tcp_frame_m'], quaternion_wxyz=f['quaternion_in_tcp_frame_wxyz'], member_spread_deg=o['uncertainty']['member_spread_deg'],
                                 uncertain=o['abstain'], abstain=o['abstain'], flip_flag=o['flip_flag'], value=o['value'], uncertainty=o['uncertainty'], note=o['note'], seconds=round(time.time() - t, 3)))
        except Exception as e:  # noqa: BLE001
            self._send(500, dict(error='%s: %s' % (type(e).__name__, str(e)[:300]), trace=traceback.format_exc()[-800:]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, required=True); a = ap.parse_args()
    EST['e'] = IV.load('cuda'); print('loaded inhand v4 charger', flush=True); ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
