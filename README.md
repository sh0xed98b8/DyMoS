# Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models

Official code for the paper
**"Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models."**

DyMoS is a **training-free** inference-time intervention that biases the
self-attention logits of pretrained image-to-video diffusion transformers to
weaken the *reference-frame attractor*. During the first few denoising steps,
attention from later frames toward the conditioning (first) frame is reduced
by a constant log-bias `gamma`. This restores motion while preserving subject
identity and the conditioning image.

This release contains the Wan 2.2 reference implementation
(`dymos_wan22.py`). Implementations for other backbones
(CogVideoX-5B, Wan 2.1, HunyuanVideo-1.5) will be released later.

## Layout

```
DyMoS/
├── dymos_wan22.py
├── prompts/
│   └── sample.txt              
├── first_frames/
│   ├── cyclists_burning_man.png
│   ├── boy_jumping_mud.jpg
│   ├── man_mountain_bike.jpg
│   └── wet_dog.png
├── LICENSE
└── README.md
```

## Environment

`dymos_wan22.py` wraps the official Wan 2.2 I2V-A14B inference pipeline.
Set up the environment according to the upstream Wan 2.2 repository:

- Upstream: https://github.com/Wan-Video/Wan2.2

Make sure the upstream `wan` package is importable from the directory you
run `dymos_wan22.py` in. Additional requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
Pillow, numpy
```

Download the Wan 2.2 I2V-A14B checkpoint and pass its path via
`--ckpt_dir`.

## Paper configuration

DyMoS introduces two hyperparameters: the bias strength `gamma` and the
number of early steps to which the bias is applied
(`K = apply_until_step`). With `T = 40` total denoising steps for
Wan 2.2, the paper uses:

| Backbone | T  | gamma | K | lambda = K / T |
|---|---|---|---|---|
| Wan 2.2  | 40 | 0.6   | 8 | 0.20 |

## Usage

Single-prompt generation:

```bash
python dymos_wan22.py \
    --ckpt_dir <path/to/Wan2.2-I2V-A14B> \
    --size 832*480 \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 0.6 --apply_until_step 8 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --out_dir output/wan22
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
python dymos_wan22.py \
    --ckpt_dir <path/to/Wan2.2-I2V-A14B> \
    --size 832*480 \
    --prompt_file prompts/sample.txt \
    --gamma 0.6 --apply_until_step 8 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --out_dir output/wan22
```
