# SANA-Video 2B

> Native text-to-video and image-to-video generation at 480p and 720p,
> plus a validated Diffusers-adapter compatibility path

## Summary

- Vendor: NVIDIA
- Model: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`,
  `Efficient-Large-Model/SANA-Video_2B_720p_diffusers`
- Task: Text-to-video and image-to-video
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use the native `SanaVideoPipeline` for T2V and
`SanaImageToVideoPipeline` for I2V. Both native pipelines support the 480p
and 720p checkpoints. Use `--diffusion-load-format diffusers` when you need
the black-box Diffusers compatibility baseline; adapter T2V and I2V are
validated at both resolutions.

The native pipeline loads the 480p checkpoint through
`DistributedAutoencoderKLWan` and the 720p checkpoint through
`DistributedAutoencoderKLLTX2Video`. These are vLLM-Omni distributed wrappers
around the corresponding Diffusers autoencoders, not independent VAE
implementations. The denoising loop also intentionally loads Diffusers'
`DPMSolverMultistepScheduler` from the checkpoint to preserve its scheduler
configuration and numerical behavior.

## References

- Upstream project: <https://github.com/NVlabs/Sana>
- Model cards:
  [480p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers),
  [720p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p_diffusers)
- Diffusers documentation: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- Online serving guides:
  [Text-to-Video](../../docs/user_guide/examples/online_serving/text_to_video.md),
  [Image-to-Video](../../docs/user_guide/examples/online_serving/image_to_video.md)
- Offline T2V example:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Offline I2V example:
  [`examples/offline_inference/image_to_video/image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py)
- Support discussion: [vLLM-Omni issue #5432](https://github.com/vllm-project/vllm-omni/issues/5432)

## Hardware Support

## GPU

### 1x RTX 5090 32GB

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.3
- Driver / runtime: NVIDIA driver 580.95.05; PyTorch 2.11.0+cu130
- Diffusers: 0.38.0
- vLLM version: 0.26.0
- vLLM-Omni version or commit: PR #5508, commit `22037901`

#### Command

##### Native text-to-video inference

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  --model-class-name SanaVideoPipeline \
  --prompt "A cat walking on the grass, facing the camera." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 704 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_720p.mp4
```

For 480p, select `SANA-Video_2B_480p_diffusers` and use
`--height 480 --width 832`.

##### Native image-to-video inference

SANA checkpoints declare `SanaVideoPipeline` in `model_index.json`, so I2V
must be selected explicitly with `--model-class-name
SanaImageToVideoPipeline`.

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --model-class-name SanaImageToVideoPipeline \
  --image input.jpg \
  --prompt "A cat turns toward the camera with smooth, natural motion." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 6 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 --seed 42 --output sana_video_i2v_480p.mp4
```

The same pipeline class supports the 720p checkpoint through vLLM-Omni's
distributed LTX-2 VAE wrapper; use `--height 704 --width 1280`.

For online I2V serving:

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

##### Native online serving

```bash
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/text_to_video/run_server_sana_video.sh

bash examples/online_serving/text_to_video/run_curl_sana_video.sh
```

To run the black-box compatibility backend for T2V, replace the server script
with `run_server_sana_video_diffusers.sh`. The same `/v1/videos` request
works; `num_frames` is adapted to Diffusers' `frames` argument. The script
selects `TORCH_SDPA` because SANA-Video uses an attention mask that the
AITER-backed Diffusers attention path does not accept.

##### Diffusers-adapter image-to-video serving

The validated I2V adapter commands are:

```bash
# 480p
MODEL=Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh

INPUT_IMAGE=input.jpg OUTPUT_PATH=sana_video_i2v_adapter.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh

# 720p
MODEL=Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
  bash examples/online_serving/image_to_video/run_server_sana_video_diffusers.sh

INPUT_IMAGE=input.jpg WIDTH=1280 HEIGHT=704 \
  OUTPUT_PATH=sana_video_i2v_adapter_720p.mp4 \
  bash examples/online_serving/image_to_video/run_curl_sana_video.sh
```

#### Verification

Check the encoded 720p output metadata after running a generation command:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 sana_video_720p.mp4
```

The standard 720p request above should report:

```text
width=1280
height=704
r_frame_rate=16/1
nb_frames=81
```

For 480p, expect `width=832` and `height=480`. To validate the documented
arguments and SANA-specific example wiring without loading the full model,
run:

```bash
.venv/bin/python -m pytest -q \
  tests/examples/offline_inference/test_sana_video_documented_commands.py
```

The automated serving matrix covers both checkpoint variants:

| Backend | 480p T2V | 720p T2V | 480p I2V | 720p I2V |
|---|---|---|---|---|
| Native vLLM-Omni | Validated | Validated | Validated | Validated |
| Diffusers adapter | Validated | Validated | Validated | Validated |

Use the native `SanaVideoPipeline` and `SanaImageToVideoPipeline` for the
primary SANA execution paths. The Diffusers adapter is retained as a
validated compatibility/reference backend.

#### Notes

- Memory usage: the native 720p, 81-frame, 50-step request took 33.56 seconds
  and reserved 23.58 GiB peak GPU memory. The corresponding
  Diffusers-adapter I2V request took about 36.5 seconds and peaked at 25.6
  GiB. A native 480p, 9-frame, one-step smoke run reserved 21.13 GiB.
- Key flags: select I2V explicitly with `--model-class-name
  SanaImageToVideoPipeline`; SANA checkpoint `model_index.json` files declare
  the T2V class. Pass `motion_score` through `--extra-body`. The supplied
  Diffusers-adapter server scripts select `TORCH_SDPA` because the AITER-backed
  path does not accept SANA's attention mask.
- Output profile: 81 frames at 16 FPS is the standard checkpoint profile and
  produces approximately five seconds of video. Minute-scale generation
  requires the separate LongSANA/LongLive block-autoregressive workflow.
- Backend boundary: the native pipelines and Transformer are owned by
  vLLM-Omni. The 480p Wan VAE and 720p LTX-2 VAE run through vLLM-Omni
  distributed wrappers derived from the corresponding Diffusers VAE classes.
  The denoising loop intentionally retains the checkpoint-compatible
  Diffusers `DPMSolverMultistepScheduler`.
- Known limitations:
  - Sequence/tensor/CFG parallelism, Cache-DiT, TeaCache, and step execution
    are not validated for the native pipeline.
  - The Diffusers backend is a compatibility path and does not provide native
    vLLM-Omni parallelism or continuous batching.
  - Native describes pipeline and Transformer ownership, not a zero-Diffusers
    dependency guarantee.
