"""
DeepDream (PyTorch) — multi-octave, with detail-reinjection pyramid and video support.

Runs gradient ascent on an input image or video to amplify the activations of an
intermediate layer of a pretrained GoogLeNet, producing the DeepDream effect.

Usage:
    python deepdream.py --input photo.jpg
    python deepdream.py --input clip.mp4 --steps 30
"""

import argparse
import math
import os
import ssl

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import GoogLeNet_Weights

# Allow torchvision to download pretrained weights in environments with strict SSL.
ssl._create_default_https_context = ssl._create_unverified_context

# Use a non-GUI backend so the script runs headless (no windows pop up).
matplotlib.use("Agg")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation constants used by the pretrained model.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model():
    """Load a pretrained GoogLeNet and the layer we'll dream on."""
    model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    target_layer = model.inception4d
    return model, target_layer


def deprocess(tensor):
    """Convert a normalised model tensor back to a 0..1 RGB image."""
    tensor = tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    tensor = tensor * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(tensor, 0, 1)


def deepdream_step(img, model, layer, step_size=0.02, jitter=16, channel=None, blend=0.05):
    """One gradient-ascent step: amplify the chosen layer's activations."""
    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    img = torch.roll(img, shifts=(ox, oy), dims=(2, 3)).detach()
    img.requires_grad_(True)
    img.retain_grad()

    act = None

    def hook_fn(_, __, output):
        nonlocal act
        act = output if channel is None else output[:, channel, :, :]

    hook = layer.register_forward_hook(hook_fn)
    model(img)

    loss = act.norm() if channel is None else act.mean()
    loss.backward()

    grad = img.grad
    grad = grad / (grad.std() + 1e-8)  # more stable than mean-based normalisation

    with torch.no_grad():
        img += step_size * grad
        img = torch.roll(img, shifts=(-ox, -oy), dims=(2, 3))
        # Slight blend back toward a clamped copy to avoid colour blow-out.
        img = (1 - blend) * img + blend * img.detach().clamp(-1, 1)

    hook.remove()
    img.grad = None
    return img.detach()


def deepdream(model, base_img, layer, octaves=(1.6, 1.4, 1.2, 1.0, 0.8),
              steps_per_octave=45, channel=None):
    """Run DeepDream across several scales (octaves) and return to original size."""
    img = base_img.clone()

    for scale in octaves:
        size = [int(base_img.shape[2] * scale), int(base_img.shape[3] * scale)]
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=True)

        for i in range(steps_per_octave):
            step_frac = 1 - i / steps_per_octave
            img = deepdream_step(
                img, model, layer,
                step_size=0.03 * step_frac,  # adaptive step size
                jitter=24, blend=0.03, channel=channel,
            )

    # Return to the original resolution.
    img = F.interpolate(
        img, size=[base_img.shape[2], base_img.shape[3]],
        mode="bilinear", align_corners=True,
    )
    return img


def _pyramid_sizes(w, h, effect_resolution=224, ratio=1.4):
    """Build an increasing sequence of sizes from effect_resolution up to (w, h)."""
    maxdim = max(w, h)
    if effect_resolution is None or effect_resolution <= 0:
        return [(w, h)]

    s = effect_resolution / float(maxdim)
    if s >= 1.0:
        return [(w, h)]

    sizes = []
    cur_w = max(32, int(round(w * s)))
    cur_h = max(32, int(round(h * s)))
    sizes.append((cur_w, cur_h))

    while sizes[-1] != (w, h):
        next_w = min(w, int(round(sizes[-1][0] * ratio)))
        next_h = min(h, int(round(sizes[-1][1] * ratio)))
        if next_w == sizes[-1][0] and next_h == sizes[-1][1]:
            sizes.append((w, h))
        else:
            sizes.append((next_w, next_h))

    # Ensure strictly increasing and unique.
    uniq = []
    for sz in sizes:
        if not uniq or uniq[-1] != sz:
            uniq.append(sz)
    return uniq


def deepdream_pyramid_frame(frame_rgb, model, layer, steps_per_level=20,
                            effect_resolution=224, ratio=1.4):
    """DeepDream a single frame using a detail-reinjection pyramid.

    frame_rgb: H x W x 3, RGB uint8.
    """
    h, w = frame_rgb.shape[:2]
    sizes = _pyramid_sizes(w, h, effect_resolution=effect_resolution, ratio=ratio)

    prev_base = None
    prev_dream = None

    for (tw, th) in sizes:
        interp_down = cv2.INTER_AREA
        interp_up = cv2.INTER_LANCZOS4
        img_level = cv2.resize(
            frame_rgb, (tw, th),
            interpolation=interp_down if (tw < w or th < h) else interp_up,
        )

        if prev_dream is None:
            init = img_level.astype(np.float32) / 255.0
        else:
            prev_up = cv2.resize(prev_dream, (tw, th), interpolation=interp_up).astype(np.float32) / 255.0
            prev_base_up = cv2.resize(prev_base, (tw, th), interpolation=interp_up).astype(np.float32) / 255.0
            detail = np.clip(prev_up - prev_base_up, -1.0, 1.0)
            init = np.clip(img_level.astype(np.float32) / 255.0 + detail, 0.0, 1.0)

        tensor = preprocess(init).unsqueeze(0).to(device)
        dreamed_tensor = deepdream(model, tensor, layer, octaves=(1.0,),
                                   steps_per_octave=steps_per_level)
        dreamed_img = deprocess(dreamed_tensor)  # 0..1 RGB float

        prev_base = img_level.copy()
        prev_dream = (dreamed_img * 255).astype(np.uint8)

    # Ensure the final size matches the original exactly.
    if prev_dream.shape[1] != w or prev_dream.shape[0] != h:
        prev_dream = cv2.resize(prev_dream, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return prev_dream


def deepdream_video(input_path, output_path, model, layer,
                    octaves=(1.6, 1.4, 1.2, 1.0, 0.8), steps_per_octave=30,
                    effect_resolution=224, use_pyramid=True):
    """DeepDream every frame of a video and write the result to MP4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames at {fps:.2f} FPS")

    if not output_path.lower().endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    if not out.isOpened():
        print("Error: could not open MP4 writer.")
        cap.release()
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if use_pyramid:
            dreamed_rgb = deepdream_pyramid_frame(
                frame, model, layer,
                steps_per_level=max(10, steps_per_octave // 2),
                effect_resolution=effect_resolution,
            )
        else:
            tensor = preprocess(frame).unsqueeze(0).to(device)
            dreamed_tensor = deepdream(model, tensor, layer,
                                       octaves=octaves, steps_per_octave=steps_per_octave)
            dreamed_rgb = (deprocess(dreamed_tensor) * 255).astype(np.uint8)

        dreamed_frame = cv2.cvtColor(dreamed_rgb, cv2.COLOR_RGB2BGR)
        if dreamed_frame.shape[1] != target_w or dreamed_frame.shape[0] != target_h:
            dreamed_frame = cv2.resize(dreamed_frame, (target_w, target_h),
                                       interpolation=cv2.INTER_LANCZOS4)
        out.write(dreamed_frame)

        frame_count += 1
        if frame_count % 5 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Finished. Saved to: {output_path}")


def deepdream_image(input_path, output_path, model, layer, effect_resolution=224,
                    steps_per_octave=45):
    """DeepDream a single image and save the result."""
    img = cv2.imread(input_path)
    if img is None:
        print("Error: could not read image.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Process at a reduced resolution for the effect, then upscale back.
    max_dim = max(w, h)
    if effect_resolution < max_dim:
        scale = effect_resolution / float(max_dim)
        proc_w = max(32, int(round(w * scale)))
        proc_h = max(32, int(round(h * scale)))
    else:
        proc_w, proc_h = w, h

    img_small = cv2.resize(img, (proc_w, proc_h)) if (proc_w, proc_h) != (w, h) else img
    tensor = preprocess(img_small).unsqueeze(0).to(device)
    dreamed_small = deepdream(model, tensor, layer, steps_per_octave=steps_per_octave)
    dreamed_img = cv2.resize(
        (deprocess(dreamed_small) * 255).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )

    cv2.imwrite(output_path, cv2.cvtColor(dreamed_img, cv2.COLOR_RGB2BGR))
    print(f"Saved image to: {output_path}")


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepDream in PyTorch (image or video).")
    parser.add_argument("--input", required=True, help="Path to input image or video.")
    parser.add_argument("--output-dir", default="output", help="Directory to save results.")
    parser.add_argument("--steps", type=int, default=45, help="Gradient-ascent steps per octave.")
    parser.add_argument("--effect-resolution", type=int, default=224,
                        help="Resolution the effect is computed at (controls detail size).")
    parser.add_argument("--no-pyramid", action="store_true",
                        help="Disable the detail-reinjection pyramid for video.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    model, target_layer = load_model()

    name, ext = os.path.splitext(os.path.basename(args.input))
    ext = ext.lower()

    if ext in VIDEO_EXTS:
        output_path = os.path.join(args.output_dir, f"{name}_deepdream.mp4")
        deepdream_video(
            args.input, output_path, model, target_layer,
            steps_per_octave=args.steps,
            effect_resolution=args.effect_resolution,
            use_pyramid=not args.no_pyramid,
        )
    elif ext in IMAGE_EXTS:
        output_path = os.path.join(args.output_dir, f"{name}_deepdream.jpg")
        deepdream_image(
            args.input, output_path, model, target_layer,
            effect_resolution=args.effect_resolution,
            steps_per_octave=args.steps,
        )
    else:
        print(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    main()
