# segment_small (harness v2 RC, DINOv3 segmenter, coin and mahjong tiles)

Pixel regions of small or thin objects in one camera image: coins (head and wrist cameras) and mahjong tiles (head and wrist cameras), from a frozen DINOv3 image model with a small segmentation head. No motion, 0 control steps.

**Inputs.** object: coin or mahjong; camera: cam_high, cam_left_wrist or cam_right_wrist.
**Returns.** {value: {n_instances, instances [{bbox_px, centroid_px, area_px, mean_prob}]}; uncertainty: evidence and threshold; abstain; steps_used 0; note}. Pixel coordinates are in the 640x480 image of that camera.
**Evidence (pre-registered, unseen Astra episodes).** Share of visible instances found: coin head 0.80, coin wrist 0.89, mahjong head 0.92, mahjong wrist 0.93; false positives at most 0.2 % of views. The generic SAM3 locator found the coin in 2 % (head) and 22 % (wrist) of the same instances.
**Known limits.** Keys and other objects are not available. Adjacent mahjong tiles can merge into one region. Regions are 2D; a 3D position needs a second view or a known support plane.
**Cost.** 0 steps.
