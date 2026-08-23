# MiniMax-H3

MiniMax-H3 jointly generates video and stereo audio with a packed diffusion
transformer. The implementation supports text-to-audio-video (T2VA), optional
first-frame conditioning (I2VA), first/last-frame conditioning (FL2VA), and
image/video/audio reference conditioning (Ref2VA).

## Model files

Pass the MiniMax-H3 components separately:

- `--diffusion-model`: MiniMax-H3 diffusion transformer
- `--vae`: MiniMax-H3 video VAE
- `--audio-vae`: MiniMax-H3 audio VAE
- `--llm`: a supported Qwen3-VL text encoder checkpoint
- `--llm_projection`: optional ClipProj weights for a smaller text encoder

The original text encoder is the MiniMax-H3 Qwen3-VL-32B variant, truncated to
50 language layers and exported without the final language-model normalization.
It can be used directly without `--llm_projection`.

Qwen3-VL-4B is also supported with the learned
`mmh3-4b-ClipProj-v3.1.safetensors` projection. The file selects its calibrated
intermediate layer through Safetensors metadata, maps the 2560-wide hidden
states to MiniMax-H3's 5120-wide conditioning, and restores the calibrated
attention-sink token. Language layers after the calibrated tap are not loaded.
This is a learned approximation of the original 32B conditioning, so prompt
and reference fidelity can differ.

In both cases, the Qwen3-VL vision tower and its three DeepStack mergers must be
present for image and video inputs. If the vision tower is stored separately,
pass it with `--llm_vision`.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

- Download minimax_h3_fl2va/minimax_h3_ref2va
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Optional Qwen3-VL-4B + ClipProj alternative
    - Qwen3-VL-4B INT8: https://huggingface.co/Winnougan/Comfy-Qwen3-VL-INT8
    - ClipProj v3.1: https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae
- Download audio vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae

## Text-to-audio-video

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten’s energetic surfing." --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

To use the 4B encoder instead, replace the 32B `--llm` argument with:

```sh
--llm ..\models\text_encoders\qwen3vl_4b_int8_convrot.safetensors --llm_projection ..\models\text_encoders\mmh3-4b-ClipProj-v3.1.safetensors
```

The same encoder and projection work with the FL2VA and Ref2VA diffusion
models. The C API exposes this path through
`new_sd_ctx_with_llm_projection()`.

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

## First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "a lovely cat" -i ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

## Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Use the cat from <Picture 1> as the main character. Keep the cat’s appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat’s face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat’s head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat’s fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat’s identity unchanged." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video is a directory of image frames sorted lexicographically and is
treated as 24 fps. Repeated `--ref-video-audio` WAV files are paired by index
with repeated `--ref-video` inputs. WAV PCM (8/16/24/32-bit) and 32/64-bit
floating-point samples are accepted; audio is converted to stereo 32 kHz by the
pipeline.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images keep their aspect ratio and are only downscaled when their
pixel area exceeds the requested generation canvas.

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
`ref_audios` in `sd_vid_gen_params_t`. Each `sd_ref_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.
