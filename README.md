# DeepDream (PyTorch)

A PyTorch implementation of the DeepDream algorithm that amplifies the
patterns a convolutional neural network "sees" in an image or video. It runs
gradient ascent on an input to maximise the activations of a chosen layer in a
pretrained GoogLeNet, producing the characteristic hallucinatory DeepDream
effect. Works on both still images and video.

## What it does

- Loads a pretrained **GoogLeNet** and hooks into an intermediate inception
  layer to read its activations.
- Performs **gradient ascent** on the input to amplify those activations, with
  gradient normalisation and jitter for stability.
- Uses **multi-octave processing** (running the effect at several scales) so
  detail appears at multiple resolutions.
- Implements a **detail-reinjection pyramid** so structure is carried across
  scales rather than washed out.
- Processes **video** frame by frame and writes the result back out to MP4.

## Tech

Python, PyTorch, torchvision, OpenCV, NumPy.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run on an image:

```bash
python deepdream.py --input path/to/photo.jpg
```

Run on a video:

```bash
python deepdream.py --input path/to/clip.mp4
```

Output is saved to an `output/` folder next to the script. The script
auto-detects whether the input is an image (`.jpg`, `.jpeg`, `.png`) or a video
(`.mp4`, `.mov`, `.avi`, `.mkv`).

Optional flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--input` | Path to the input image or video (required) | — |
| `--output-dir` | Where to save results | `output/` |
| `--steps` | Gradient-ascent steps per octave | `45` |
| `--effect-resolution` | Resolution the effect is computed at (detail size) | `224` |
| `--no-pyramid` | Disable detail-reinjection pyramid for video | pyramid on |


## How it works (short version)

DeepDream picks a layer inside a trained image classifier and asks: "what would
make this layer fire more strongly?" By computing the gradient of that layer's
activation with respect to the input pixels and stepping the image in that
direction, it exaggerates whatever the layer already partially detects, edges,
textures, eyes, fur, into vivid, repeating patterns. Doing this across multiple
scales (octaves) and reinjecting fine detail between scales is what gives the
output its fractal, all over richness.
