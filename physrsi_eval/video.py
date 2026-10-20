"""Episode videos from the per-step camera archive: cam_high | cam_left_wrist | cam_right_wrist, 25 Hz.

One frame per native control step (the simulator records all three 640x480 views after
every step), encoded with libx264 (preset fast, CRF 20, yuv420p) through imageio-ffmpeg.
"""
import re
from pathlib import Path

import numpy as np

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
FPS = 25


def frames(sim_output):
    folder = Path(sim_output) / 'observations'
    steps = sorted({int(m.group(1)) for p in folder.glob('*-cam_high.png')
                    if (m := re.match(r'(\d+)-cam_high\.png$', p.name))})
    from PIL import Image
    for step in steps:
        yield np.concatenate([np.asarray(Image.open(folder / f'{step:06d}-{cam}.png').convert('RGB'))
                              for cam in CAMERAS], axis=1)


def render(sim_output, target, fps=FPS):
    import imageio.v2 as imageio
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with imageio.get_writer(str(target), fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=2,
                            output_params=['-preset', 'fast', '-crf', '20', '-movflags', '+faststart']) as writer:
        for frame in frames(sim_output):
            writer.append_data(frame)
            count += 1
    if count == 0:
        target.unlink(missing_ok=True)
        raise FileNotFoundError(f'no recorded frames under {sim_output}/observations')
    return count


def render_chain(chain, out_dir):
    """One video per round of a chain (K0 repeats included when present)."""
    chain, out_dir = Path(chain), Path(out_dir)
    made = []
    for sim in sorted(chain.glob('K[0-9][0-9]/sim')) + sorted(chain.parent.glob('K0/r[0-9]/sim')):
        name = sim.parent.name if sim.parent.parent == chain else 'K0-' + sim.parent.name
        target = out_dir / f'{chain.parent.name}-{chain.name}-{name}.mp4'
        try:
            made.append((target, render(sim, target)))
        except FileNotFoundError:
            continue
    return made
