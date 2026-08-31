## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).

Run by adding `--diffusion-fa` to the arguments and watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use params backend to reduce VRAM or RAM usage.

`--params-backend` controls where model parameters are kept. If it is not set, parameters use the same backend as `--backend`, so a GPU runtime backend also keeps parameters in VRAM.

Use CPU params to reduce VRAM usage:

```shell
--backend cuda0 --params-backend cpu
```

This keeps model weights in system RAM and moves them to the runtime backend when needed. In the example CLI/server, `--offload-to-cpu` is a compatibility shortcut that prepends `*=cpu` to `--params-backend` before creating the context, so explicit module assignments can still override it:

```shell
--offload-to-cpu --params-backend te=disk
```

Use disk params to reduce both VRAM and RAM usage:

```shell
--backend cuda0 --params-backend disk
```

This reloads parameters from the model file on demand and releases them after use. It has the lowest memory residency, but can be slower because weights must be read again. With layer streaming on Linux, compatible tensors in regular, uncompressed model files use aligned `O_DIRECT` reads through reusable pinned host buffers and asynchronous device copies. Other tensors and platforms fall back to buffered reads. `disk` is never selected implicitly; set it explicitly when RAM usage matters more than reload cost.

Per-module assignments can target only the largest modules:

```shell
--backend cuda0 --params-backend diffusion=disk,te=cpu,vae=cpu
```

See [backend selection](./backend.md) for full syntax.

## Run models that don't fit in VRAM (layer streaming).

`--offload-to-cpu` alone keeps every parameter in system RAM and stages it to the runtime backend on first use, then leaves it resident there. If the diffusion model is larger than the runtime backend's free memory (e.g. Flux dev at bf16 on an 8 GiB GPU), that residency stops fitting during the sampling loop and generation fails. The following flags make it fit by trading a small amount of speed for room:

- `--max-vram <GiB>` sets a VRAM budget the graph-cut segmenter respects. It cuts each forward pass into segments sized to fit the budget, running them in sequence and freeing intermediate activations between them. The default is `-1`: the budget is detected automatically from free VRAM with ~1 GiB headroom. A positive value sets an explicit cap and `0` disables segmentation.
- `--stream-layers` streams the diffusion model's transformer blocks one at a time. Each block's parameters are copied from the CPU parameter backend or reloaded from the disk parameter source just before it runs, then evicted when the residency budget is reached. Prefetching hides as much of the transfer latency as the current segment's compute time permits. This flag requires `--params-backend diffusion=cpu` (including `--offload-to-cpu`) or `--params-backend diffusion=disk`; a warning is logged and the flag is ignored otherwise.
- `--resident-layers <N|auto>` sets the maximum number of leading parameter-bearing graph-cut segments kept resident (default: `-1`; `auto` is an alias for `-1`). `-1` uses as many as the VRAM budget permits, `0` keeps none, and a positive `N` keeps up to `N`.
- `--layer-prefetch-depth <N>` sets the maximum number of future parameter-bearing graph-cut segments copied through a separate transfer backend or queue when supported while the active segment computes (default: `0`). `0` disables asynchronous prefetching, `1` overlaps the next segment, and larger values provide deeper lookahead when the VRAM budget permits.
- `--stream-vram-safety <MiB>` controls the additional 512 MiB margin inside the layer-streaming budget (default: `512`; accepted values: `512` or `0`). Set it to `0` to disable this extra margin when the outer `--max-vram` headroom is sufficient for the workload.
- `--stream-layer-pool` keeps one contiguous backend allocation alive for the sampling loop instead of repeatedly allocating buffers for segment-private streamed parameters (disabled by default; its presence enables it). The allocation is divided into fixed-capacity slots, each sized for the largest private parameter set of any streamed segment. This trades additional reserved VRAM for fewer backend allocations and frees during inference.

These controls require `--stream-layers`; the default automatic VRAM budget enables them without an explicit `--max-vram`. Prefetch is budgeted before residency; requested limits are reduced as needed, a positive VRAM cap is never exceeded to force an optional allocation, and the active segment is never evicted. Residents persist only across repeated sampling steps, and stale graph state is released automatically. An explicit non-negative residency limit uses an unmerged plan, which may add dispatch overhead. With `--resident-layers 0 --layer-prefetch-depth 1 --stream-layer-pool`, the active and next segments alternate between two reusable slots when the budget permits.

The direct disk path uses four 64 MiB pinned staging buffers and Linux native AIO by default. Nearby tensor ranges are read as continuous windows when the segment is large enough to keep at least two full waves of AIO requests queued; shorter groups retain per-tensor reads. Set `SD_DIRECT_STORAGE_BUFFER_MIB` to an integer from 4 through 1024 and `SD_DIRECT_STORAGE_BUFFER_COUNT` from 1 through 8 to tune the startup/memory/per-segment-I/O tradeoff for the storage device. If native AIO is unavailable, the loader automatically retains the same `O_DIRECT` staging path with synchronous `pread`; set `SD_DIRECT_STORAGE_AIO=0` to select that fallback explicitly, or `SD_DIRECT_STORAGE=0` to force fully buffered reads. These environment variables affect only streamed disk parameters.

The pool always attempts to allocate the one slot required to execute the largest segment, even when that slot exceeds the effective `--max-vram` budget; the budget limits additional slots, prefetch, and residency. If the requested contiguous allocation fails, the slot count is reduced without changing the slot size. Failure to allocate even one slot stops inference, while failure of asynchronous lookahead only disables prefetch: current segments continue to load synchronously through the same pool, which remains allocated until sampling ends.

These flags stack. The recommended shape for "biggest model my card can host":

```shell
sd-cli --diffusion-model flux1-dev.safetensors ... \
       --offload-to-cpu --max-vram -1 --stream-layers \
       --resident-layers auto --layer-prefetch-depth 1
```

To avoid keeping diffusion weights in system RAM, use the disk source with a reusable two-slot pool:

```shell
sd-cli --diffusion-model model.safetensors ... \
       --params-backend diffusion=disk,te=cpu,vae=cpu \
       --stream-layers --resident-layers 0 --layer-prefetch-depth 1 \
       --stream-layer-pool
```

- `--offload-to-cpu`: params in RAM, staged as needed.
- `--max-vram -1`: use most of the free VRAM as the compute budget, spare 1 GiB headroom, let the graph-cut segmenter split each forward pass to fit.
- `--stream-layers`: on top of the segmenter, stream individual transformer blocks so their weights don't all need to be resident at once.
- `--resident-layers auto`: use the remaining budget for a leading resident prefix.
- `--layer-prefetch-depth 1`: prepare the next parameter-bearing segment concurrently with the current segment's computation.

Ordered from fastest to smallest-VRAM: no flags → `--offload-to-cpu` → `--offload-to-cpu --max-vram <N>` → `--offload-to-cpu --max-vram <N> --stream-layers`. Each step down costs a few percent of throughput to buy more room; combined they can run models roughly 3-4x larger than the raw VRAM would allow.

## Use quantization to reduce memory usage.

[quantization](./quantization_and_gguf.md)
