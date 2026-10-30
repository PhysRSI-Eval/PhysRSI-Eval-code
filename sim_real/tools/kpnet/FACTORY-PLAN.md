# Keypoint tool factory (post-training line ip4; user 11:5xZ "v6 补充" + TASK-CATEGORIES C1 -> C3 -> C2)
One generic pipeline. The finepose tools are copied read-only; their sha256 is in FINEPOSE-REUSE.sha256.
1. **CAD**: src/export_cad.py NAME=REL ... -> cad/<name>.{obj,npz,json}. Exported so far: charger, socket, coin.
2. **Keypoint table** (constants, in the object frame), per instance:
   - plug_in_charger:
     - charger: prong tips (+-0.0055, -0.0152, 0), prong bases (+-0.0055, 0, 0), body corner (0.021, 0.0479, 0.0135);
     - socket (sim frame of socket_0_2, as truth_outlets): 3 outlets x 2 holes at (OFFS +- 0.0055, OUT_Y, z_top). K = 11.
   - insert_tubes: tube tip / tube top + rack hole centres (next).
   - deposit_coin: coin rim points + piggy-bank slot ends and centre (next).
3. **State generator, physical (no teleport)**: native reset -> scripted grasp -> carry -> idle-arm side views. Finepose park():
   azimuth sweep, 0.12-0.18 m, aim jitter; half the views on the held object, half on the target. Also the holding wrist and the head.
   Per view: RGB + sim depth (gpd_server_ext) + FK cameras + truth poses. src/kp_charger_gen.py.
4. **Labels**: CAD points at truth poses projected with the FK cameras. Visibility = in image AND sim depth at the pixel >= point
   depth - 3 mm. The fingers are modelled by the real depth, which finepose lacked.
5. **Training**: kpnet3 recipe unchanged, one net per camera type (idle-side wrist, holding wrist). Split by episode; layouts 3/8/13 are
   TEST. Rung 1: DEV-chosen checkpoint; TEST median px and triangulated 3-D error.
6. **API**: triangulate(views) -> 3-D keypoints; relative_error(object points, target points) -> dxy, dz, dyaw. These are the inputs
   to the capability line's relative visual-servo executor. Registered through register_kp_tool() after rung 1.
7. **KPT comparison**: the same data and split, with KPT (DINOv3 + CAD queries) instead of KPNet. Pixel and 3-D error on identical
   frames.

## Status 11:5xZ
- charger: 144 jobs (18 layouts x 8 seeds), 10 workers (host1 GPU1-4, host2 GPU2/4). The first 4 jobs give 2-8 usable views each.
  The idle-arm views fail when the target is too far for that arm; those views are skipped and recorded.

## Queue (dispatcher 13:0xZ)
1. plug_in_charger. Co-visible hover data running, K = 14 per the capability spec (to-ip4.md).
2. C2 long objects, for the C2 line (experiments/c2-grasp-20261029/) grasp rules:
   - hammer, wrench, bottle (put_bottles_into_dustbin / pour);
   - keypoints: head end, handle end, CAD centre of mass. Deploy recovers the long-axis direction AND which end is which.
3. insert_tubes: tube tip / top + rack holes.
4. deposit_coin: coin rim + slot ends / centre.
