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


def preview(sim_output, target, camera='cam_high', width=400, seconds=12, fps=10, caption=None, quality=60):
    """Small looping time-lapse (animated WebP, or GIF if `target` ends in .gif) of one camera."""
    from PIL import Image, ImageDraw
    folder = Path(sim_output) / 'observations'
    steps = sorted(int(p.name[:6]) for p in folder.glob(f'*-{camera}.png'))
    if not steps:
        raise FileNotFoundError(f'no recorded frames under {folder}')
    count = min(len(steps), seconds * fps)
    picks = [steps[round(i * (len(steps) - 1) / max(count - 1, 1))] for i in range(count)]
    frames = []
    for step in picks:
        image = Image.open(folder / f'{step:06d}-{camera}.png').convert('RGB')
        image = image.resize((width, round(image.height * width / image.width)))
        if caption:
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, image.width, 18], fill=(0, 0, 0))
            draw.text((5, 3), f'{caption}  step {step}', fill=(255, 255, 255))
        frames.append(image)
    frames += [frames[-1]] * fps  # hold the final state for one second
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == '.gif':
        frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
        frames[0].save(target, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)
    else:
        frames[0].save(target, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0,
                       quality=quality, method=6)
    return len(frames)
