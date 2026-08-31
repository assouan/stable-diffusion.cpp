#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "core/cuda_peer_access.h"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"

namespace MiniMaxH3 {

    constexpr size_t H3_GRAPH_SIZE       = 524288;
    constexpr size_t H3_TENSOR_CAPACITY  = 655360;
    constexpr size_t H3_BLOCK_STREAM_GRAPH_SIZE      = 8192;
    constexpr size_t H3_BLOCK_STREAM_TENSOR_CAPACITY = 16384;
    constexpr float FRAME_RESCALE        = 5.f / 3.f;
    constexpr float VISUAL_COND_TIMESTEP = 0.999f;

    enum class BlockStreamWarmup {
        NONE,
        LAYER,
        STEP,
    };

    struct Config {
        int64_t hidden_size              = 5376;
        int64_t num_layers               = 50;
        int64_t execution_layers         = 50;
        int64_t token_refiner_num_layers = 2;
        int64_t num_attention_heads      = 56;
        int64_t attention_head_dim       = 128;
        int64_t ffn_hidden_size          = 14336;
        int64_t mlp_chunk_size           = 4096;
        int64_t attention_chunk_size     = 4096;
        int64_t video_latent_channels    = 24;
        int64_t audio_latent_channels    = 32;
        int64_t text_dim                 = 5120;
        int64_t timestep_input_dim       = 256;
        int64_t time_embed_hidden_size   = 5376;
        int64_t time_embed_dim           = 2688;
        int64_t rope_inv_freq_len        = 16;
        int64_t adaln_curve_grid         = 0;
        int patch_t                      = 1;
        int patch_h                      = 2;
        int patch_w                      = 2;
        float norm_eps                   = 1e-5f;
        float qk_norm_eps                = 1e-5f;
        float final_norm_eps             = 1e-5f;
        float attention_sparsity         = 0.0f;
        bool sequence_parallel            = false;
        ggml_type activation_storage_type = GGML_TYPE_BF16;
        bool block_stream_enabled          = false;
        BlockStreamWarmup block_stream_warmup = BlockStreamWarmup::NONE;
        int64_t block_stream_chunk_size    = 1024;
        int block_stream_stage_layers      = 1;
        int block_stream_buffers           = 3;
        bool block_stream_trace            = false;
        std::string block_stream_error;

        bool uses_adaln_curves() const {
            return adaln_curve_grid > 0;
        }

        static int64_t count_blocks(const String2TensorStorage& tensors,
                                    const std::string& prefix) {
            std::set<int> indices;
            for (const auto& [name, _] : tensors) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t begin = prefix.size();
                size_t end   = name.find('.', begin);
                if (end != std::string::npos) {
                    indices.insert(std::atoi(name.substr(begin, end - begin).c_str()));
                }
            }
            return static_cast<int64_t>(indices.size());
        }

        static Config detect_from_weights(const String2TensorStorage& tensors,
                                          const std::string& prefix) {
            Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = tensors.find(prefix + "." + suffix);
                return it == tensors.end() ? nullptr : &it->second;
            };

            if (const auto* weight = find("video_patch_proj.weight")) {
                config.video_latent_channels = weight->ne[0] / 4;
                config.hidden_size           = weight->ne[1];
            }
            if (const auto* weight = find("audio_patch_proj.weight")) {
                config.audio_latent_channels = weight->ne[0];
            }
            config.num_layers               = count_blocks(tensors, prefix + ".blocks.");
            config.execution_layers         = config.num_layers;
            config.token_refiner_num_layers = count_blocks(tensors, prefix + ".token_refiner.blocks.");
            if (const auto* weight = find("blocks.0.attn.q_norm.weight")) {
                config.attention_head_dim = weight->ne[0];
            }
            if (const auto* weight = find("blocks.0.attn.qkv_proj.weight")) {
                config.num_attention_heads = weight->ne[1] / (3 * config.attention_head_dim);
            }
            if (const auto* weight = find("blocks.0.mlp.fc1.weight")) {
                config.ffn_hidden_size = weight->ne[1] / 2;
            }
            if (const auto* weight = find("condition_proj.weight")) {
                config.text_dim = weight->ne[0];
            }
            if (const auto* table = find("adaln_t_table")) {
                config.time_embed_dim   = table->ne[0];
                config.adaln_curve_grid = table->ne[1];
            } else {
                if (const auto* weight = find("time_embedder.proj_in.weight")) {
                    config.timestep_input_dim     = weight->ne[0];
                    config.time_embed_hidden_size = weight->ne[1];
                }
                if (const auto* weight = find("time_embedder.proj_out.weight")) {
                    config.time_embed_dim = weight->ne[1];
                }
            }
            if (const auto* inv_freq = find("rope.inv_freq")) {
                config.rope_inv_freq_len = inv_freq->ne[0];
            }

            LOG_DEBUG("minimax_h3: layers=%" PRId64 ", hidden=%" PRId64 ", heads=%" PRId64
                      ", head_dim=%" PRId64 ", ffn=%" PRId64 ", adaln_curve=%" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.attention_head_dim,
                      config.ffn_hidden_size,
                      config.adaln_curve_grid);
            return config;
        }
    };

    static float time_shift_sigma(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        return to_shift * base / (1.f + (to_shift - 1.f) * base);
    }

    static float time_shift_slope(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        float a    = 1.f + (from_shift - 1.f) * base;
        float b    = 1.f + (to_shift - 1.f) * base;
        return to_shift * a * a / (from_shift * b * b);
    }

    struct TimeEmbedder : public GGMLBlock {
        TimeEmbedder(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
            blocks["proj_in"]  = std::make_shared<Linear>(input_dim, hidden_dim, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(hidden_dim, output_dim, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            return proj_out->forward(ctx, ggml_silu(ctx->ggml_ctx, proj_in->forward(ctx, x)));
        }
    };

    static ggml_tensor* concat_balanced(ggml_context* ctx,
                                        std::vector<ggml_tensor*> tensors,
                                        int dim) {
        GGML_ASSERT(!tensors.empty());
        while (tensors.size() > 1) {
            std::vector<ggml_tensor*> next;
            next.reserve((tensors.size() + 1) / 2);
            for (size_t i = 0; i < tensors.size(); i += 2) {
                next.push_back(i + 1 < tensors.size()
                                   ? ggml_concat(ctx, tensors[i], tensors[i + 1], dim)
                                   : tensors[i]);
            }
            tensors = std::move(next);
        }
        return tensors.front();
    }

    struct MLP : public UnaryBlock {
        int64_t chunk_size;

        MLP(int64_t hidden_size,
            int64_t ffn_hidden_size,
            int64_t chunk_size)
            : chunk_size(chunk_size) {
            blocks["fc1"] = std::make_shared<Linear>(hidden_size, ffn_hidden_size * 2, false, false, true, 1.f / 128.f);
            blocks["fc2"] = std::make_shared<Linear>(ffn_hidden_size, hidden_size, false, false, true, 1.f / 128.f);
        }

        ggml_tensor* forward_chunk(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
            auto uv  = fc1->forward(ctx, x);
            return fc2->forward(ctx, ggml_glu(ctx->ggml_ctx,
                                              uv,
                                              GGML_GLU_OP_SWIGLU,
                                              false));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            if (chunk_size <= 0 || x->ne[1] <= chunk_size) {
                return forward_chunk(ctx, x);
            }

            std::vector<ggml_tensor*> outputs;
            outputs.reserve(static_cast<size_t>((x->ne[1] + chunk_size - 1) / chunk_size));
            for (int64_t start = 0; start < x->ne[1]; start += chunk_size) {
                int64_t end = std::min(start + chunk_size, x->ne[1]);
                auto output = forward_chunk(ctx,
                                            ggml_ext_slice(ctx->ggml_ctx,
                                                           x,
                                                           1,
                                                           start,
                                                           end));
                ggml_format_name(output, "h3.mlp.chunk.%lld", (long long) (start / chunk_size));
                outputs.push_back(output);
            }
            auto output = concat_balanced(ctx->ggml_ctx, std::move(outputs), 1);
            ggml_set_name(output, "h3.mlp.output");
            return output;
        }
    };

    static ggml_tensor* attention_layout(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
    }

    static ggml_tensor* apply_partial_rope(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* pe) {
        int64_t rot_dim = pe->ne[2] * 2;
        GGML_ASSERT(rot_dim <= x->ne[0]);
        auto rotated = Rope::apply_rope(ctx,
                                        ggml_ext_slice(ctx, x, 0, 0, rot_dim),
                                        pe,
                                        false);
        if (rot_dim == x->ne[0]) {
            return rotated;
        }
        auto tail = attention_layout(ctx, ggml_ext_slice(ctx, x, 0, rot_dim, x->ne[0]));
        return ggml_concat(ctx, rotated, tail, 0);
    }

    struct Attention : public GGMLBlock {
        int64_t heads;
        int64_t head_dim;
        int64_t chunk_size;
        float sparsity;

        Attention(int64_t hidden_size,
                  int64_t heads,
                  int64_t head_dim,
                  float eps,
                  int64_t chunk_size,
                  float sparsity = 0.0f)
            : heads(heads), head_dim(head_dim), chunk_size(chunk_size), sparsity(sparsity) {
            int64_t inner      = heads * head_dim;
            blocks["qkv_proj"] = std::make_shared<Linear>(hidden_size, inner * 3, false);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["out_proj"] = std::make_shared<Linear>(inner, hidden_size, false);
        }

        void set_sparsity(float value) {
            sparsity = value;
        }

        ggml_tensor* forward_full(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* pe) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            int64_t sequence = x->ne[1];
            int64_t batch    = x->ne[2] * x->ne[3];
            auto qkv         = ggml_ext_chunk(ctx->ggml_ctx, qkv_proj->forward(ctx, x), 3, 0);
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv[0], head_dim, heads, sequence, batch);
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv[1], head_dim, heads, sequence, batch);
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv[2], head_dim, heads, sequence, batch);
            q                = q_norm->forward(ctx, q);
            k                = k_norm->forward(ctx, k);
            if (pe != nullptr) {
                q = apply_partial_rope(ctx->ggml_ctx, q, pe);
                k = apply_partial_rope(ctx->ggml_ctx, k, pe);
            } else {
                q = attention_layout(ctx->ggml_ctx, q);
                k = attention_layout(ctx->ggml_ctx, k);
            }
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              static_cast<int>(heads),
                                              nullptr,
                                              true,
                                              ctx->flash_attn_enabled,
                                              1.f / 128.f,
                                              sparsity);
            return out_proj->forward(ctx, out);
        }

        std::vector<ggml_tensor*> forward_chunks(
            GGMLRunnerContext* ctx,
            int64_t sequence,
            int64_t batch,
            ggml_tensor* pe,
            const std::function<ggml_tensor*(int64_t, int64_t)>& input_for_range) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            int64_t inner = heads * head_dim;
            std::vector<ggml_tensor*> keys;
            std::vector<ggml_tensor*> values;
            size_t chunk_count = static_cast<size_t>((sequence + chunk_size - 1) / chunk_size);
            keys.reserve(chunk_count);
            values.reserve(chunk_count);

            for (int64_t start = 0; start < sequence; start += chunk_size) {
                int64_t end          = std::min(start + chunk_size, sequence);
                int64_t chunk_tokens = end - start;
                auto input           = input_for_range(start, end);
                auto k               = qkv_proj->forward_output_slice(ctx,
                                                                       input,
                                                                       inner,
                                                                       inner * 2);
                k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, heads, chunk_tokens, batch);
                k                    = k_norm->forward(ctx, k);
                if (pe != nullptr) {
                    auto pe_chunk = ggml_ext_slice(ctx->ggml_ctx, pe, 3, start, end);
                    k             = apply_partial_rope(ctx->ggml_ctx, k, pe_chunk);
                } else {
                    k = attention_layout(ctx->ggml_ctx, k);
                }

                k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, chunk_tokens, heads, batch);
                k = ggml_cast(ctx->ggml_ctx,
                              ggml_ext_scale(ctx->ggml_ctx, k, 1.f / 128.f),
                              GGML_TYPE_F16);
                ggml_format_name(k, "h3.attn.k.%lld", (long long) (start / chunk_size));
                keys.push_back(k);
                auto v               = qkv_proj->forward_output_slice(ctx,
                                                                       input,
                                                                       inner * 2,
                                                                       inner * 3);
                v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, heads, chunk_tokens, batch);
                v = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
                v = ggml_cast(ctx->ggml_ctx,
                              ggml_ext_scale(ctx->ggml_ctx, v, 1.f / 128.f),
                              GGML_TYPE_F16);
                ggml_format_name(v, "h3.attn.v.%lld", (long long) (start / chunk_size));
                values.push_back(v);
            }
            auto k = concat_balanced(ctx->ggml_ctx, std::move(keys), 1);
            ggml_set_name(k, "h3.attn.k.all");
            auto v = concat_balanced(ctx->ggml_ctx, std::move(values), 1);
            ggml_set_name(v, "h3.attn.v.all");
            std::vector<ggml_tensor*> outputs;
            outputs.reserve(chunk_count);
            for (int64_t start = 0; start < sequence; start += chunk_size) {
                int64_t end          = std::min(start + chunk_size, sequence);
                int64_t chunk_tokens = end - start;
                auto input           = input_for_range(start, end);
                auto q               = qkv_proj->forward_output_slice(ctx,
                                                                       input,
                                                                       0,
                                                                       inner);
                q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, heads, chunk_tokens, batch);
                q = q_norm->forward(ctx, q);
                if (pe != nullptr) {
                    auto pe_chunk = ggml_ext_slice(ctx->ggml_ctx, pe, 3, start, end);
                    q             = apply_partial_rope(ctx->ggml_ctx, q, pe_chunk);
                } else {
                    q = attention_layout(ctx->ggml_ctx, q);
                }
                q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, chunk_tokens, heads, batch);
                ggml_format_name(q, "h3.attn.q.%lld", (long long) (start / chunk_size));
                auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                  ctx->backend,
                                                  q,
                                                  k,
                                                  v,
                                                  static_cast<int>(heads),
                                                  nullptr,
                                                  true,
                                                  true,
                                                  1.f / 128.f,
                                                  sparsity,
                                                  true);
                auto output = out_proj->forward(ctx, out);
                ggml_format_name(output,
                                 "h3.attn.output.%lld",
                                 (long long) (start / chunk_size));
                outputs.push_back(output);
            }
            return outputs;
        }

        std::pair<ggml_tensor*, ggml_tensor*> update_kv_cache_chunk(
            GGMLRunnerContext* ctx,
            ggml_tensor* input,
            ggml_tensor* pe,
            int64_t token_start,
            ggml_tensor* key_cache,
            ggml_tensor* value_cache,
            ggml_tensor* cache_rows) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

            const int64_t chunk_tokens = input->ne[1];
            const int64_t batch        = input->ne[2] * input->ne[3];
            const int64_t inner        = heads * head_dim;
            GGML_ASSERT(chunk_tokens > 0 && cache_rows != nullptr);
            GGML_ASSERT(key_cache != nullptr && value_cache != nullptr);
            GGML_ASSERT(key_cache->type == GGML_TYPE_F16 && value_cache->type == GGML_TYPE_F16);
            GGML_ASSERT(key_cache->ne[0] == head_dim &&
                        key_cache->ne[1] >= token_start + chunk_tokens);

            auto k = qkv_proj->forward_output_slice(ctx, input, inner, inner * 2);
            k      = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, heads, chunk_tokens, batch);
            k      = k_norm->forward(ctx, k);
            if (pe != nullptr) {
                auto pe_chunk = ggml_ext_slice(ctx->ggml_ctx,
                                               pe,
                                               3,
                                               token_start,
                                               token_start + chunk_tokens);
                k = apply_partial_rope(ctx->ggml_ctx, k, pe_chunk);
            } else {
                k = attention_layout(ctx->ggml_ctx, k);
            }
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, chunk_tokens, heads, batch);
            k = ggml_cast(ctx->ggml_ctx,
                          ggml_ext_scale(ctx->ggml_ctx, k, 1.f / 128.f),
                          GGML_TYPE_F16);

            auto v = qkv_proj->forward_output_slice(ctx, input, inner * 2, inner * 3);
            v      = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, heads, chunk_tokens, batch);
            v      = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
            v      = ggml_cast(ctx->ggml_ctx,
                          ggml_ext_scale(ctx->ggml_ctx, v, 1.f / 128.f),
                          GGML_TYPE_F16);

            return {ggml_set_rows(ctx->ggml_ctx, key_cache, k, cache_rows),
                    ggml_set_rows(ctx->ggml_ctx, value_cache, v, cache_rows)};
        }

        ggml_tensor* forward_cached_query_chunk(GGMLRunnerContext* ctx,
                                                ggml_tensor* input,
                                                ggml_tensor* pe,
                                                int64_t token_start,
                                                ggml_tensor* key_cache,
                                                ggml_tensor* value_cache,
                                                int64_t valid_kv_tokens) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            const int64_t chunk_tokens = input->ne[1];
            const int64_t batch        = input->ne[2] * input->ne[3];
            const int64_t inner        = heads * head_dim;
            GGML_ASSERT(chunk_tokens > 0 && key_cache != nullptr && value_cache != nullptr);
            GGML_ASSERT(key_cache->type == GGML_TYPE_F16 && value_cache->type == GGML_TYPE_F16);
            GGML_ASSERT(valid_kv_tokens > 0 && valid_kv_tokens <= key_cache->ne[1]);

            auto q = qkv_proj->forward_output_slice(ctx, input, 0, inner);
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, heads, chunk_tokens, batch);
            q      = q_norm->forward(ctx, q);
            if (pe != nullptr) {
                auto pe_chunk = ggml_ext_slice(ctx->ggml_ctx,
                                               pe,
                                               3,
                                               token_start,
                                               token_start + chunk_tokens);
                q = apply_partial_rope(ctx->ggml_ctx, q, pe_chunk);
            } else {
                q = attention_layout(ctx->ggml_ctx, q);
            }
            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, chunk_tokens, heads, batch);

            auto visible_k = valid_kv_tokens == key_cache->ne[1]
                                 ? key_cache
                                 : ggml_ext_slice(ctx->ggml_ctx,
                                                  key_cache,
                                                  1,
                                                  0,
                                                  valid_kv_tokens);
            auto visible_v = valid_kv_tokens == value_cache->ne[1]
                                 ? value_cache
                                 : ggml_ext_slice(ctx->ggml_ctx,
                                                  value_cache,
                                                  1,
                                                  0,
                                                  valid_kv_tokens);
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              visible_k,
                                              visible_v,
                                              static_cast<int>(heads),
                                              nullptr,
                                              true,
                                              true,
                                              1.f / 128.f,
                                              0.0f,
                                              true);
            return out_proj->forward(ctx, out);
        }

        ggml_tensor* forward_cached_chunk(GGMLRunnerContext* ctx,
                                          ggml_tensor* input,
                                          ggml_tensor* pe,
                                          int64_t token_start,
                                          ggml_tensor* key_cache,
                                          ggml_tensor* value_cache,
                                          ggml_tensor* cache_rows,
                                          int64_t valid_kv_tokens) {
            auto updated = update_kv_cache_chunk(ctx,
                                                 input,
                                                 pe,
                                                 token_start,
                                                 key_cache,
                                                 value_cache,
                                                 cache_rows);
            return forward_cached_query_chunk(ctx,
                                              input,
                                              pe,
                                              token_start,
                                              updated.first,
                                              updated.second,
                                              valid_kv_tokens);
        }

        ggml_tensor* forward_sequence_parallel(GGMLRunnerContext* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* pe,
                                               ggml_type wire_type,
                                               const std::string& qkv_cut_group) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            const int64_t sequence = x->ne[1];
            const int64_t batch    = x->ne[2] * x->ne[3];
            GGML_ASSERT(batch == 1);

            if (x->type != wire_type) {
                x = ggml_cast(ctx->ggml_ctx, x, wire_type);
            }
            x = ggml_backend_meta_all_gather(ctx->ggml_ctx,
                                             ggml_cont(ctx->ggml_ctx, x),
                                             GGML_BACKEND_SPLIT_AXIS_1);
            ggml_set_name(x, "h3.sequence.attn.hidden.gathered");

            const int64_t projection_chunk_size = std::max<int64_t>(1, chunk_size);
            std::vector<ggml_tensor*> qkv_chunks;
            qkv_chunks.reserve(static_cast<size_t>((sequence + projection_chunk_size - 1) /
                                                   projection_chunk_size));
            for (int64_t start = 0; start < sequence; start += projection_chunk_size) {
                const int64_t end = std::min(start + projection_chunk_size, sequence);
                auto input_chunk = ggml_ext_slice(ctx->ggml_ctx, x, 1, start, end);
                auto qkv_chunk = ggml_cast(ctx->ggml_ctx,
                                           qkv_proj->forward(ctx, input_chunk),
                                           GGML_TYPE_F16);
                ggml_format_name(qkv_chunk,
                                 "h3.sequence.attn.qkv.%lld",
                                 (long long) (start / projection_chunk_size));
                qkv_chunks.push_back(qkv_chunk);
            }
            auto qkv = concat_balanced(ctx->ggml_ctx, std::move(qkv_chunks), 1);
            ggml_set_name(qkv, "h3.sequence.attn.qkv");
            if (!qkv_cut_group.empty()) {
                sd::ggml_graph_cut::mark_graph_cut(qkv, qkv_cut_group, "qkv");
            }
            auto qkv_parts = ggml_ext_chunk(ctx->ggml_ctx, qkv, 3, 0);

            auto q = ggml_cast(ctx->ggml_ctx, qkv_parts[0], GGML_TYPE_F32);
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, heads, sequence, batch);
            q      = q_norm->forward_out_of_place(ctx, q);
            q      = pe != nullptr ? apply_partial_rope(ctx->ggml_ctx, q, pe)
                                   : attention_layout(ctx->ggml_ctx, q);
            q      = ggml_cont(ctx->ggml_ctx, q);

            auto k = ggml_cast(ctx->ggml_ctx, qkv_parts[1], GGML_TYPE_F32);
            k      = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, heads, sequence, batch);
            k      = k_norm->forward_out_of_place(ctx, k);
            k      = pe != nullptr ? apply_partial_rope(ctx->ggml_ctx, k, pe)
                                   : attention_layout(ctx->ggml_ctx, k);
            k      = ggml_cast(ctx->ggml_ctx,
                               ggml_ext_scale(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, k), 1.f / 128.f),
                               GGML_TYPE_F16);

            auto v = ggml_cast(ctx->ggml_ctx, qkv_parts[2], GGML_TYPE_F32);
            v = ggml_reshape_4d(ctx->ggml_ctx,
                                v,
                                head_dim,
                                heads,
                                sequence,
                                batch);
            v = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_permute(ctx->ggml_ctx, v, 0, 2, 1, 3));
            v = ggml_cast(ctx->ggml_ctx,
                          ggml_ext_scale(ctx->ggml_ctx, v, 1.f / 128.f),
                          GGML_TYPE_F16);

            const int64_t attention_query_chunk_size = sparsity > 0.0f
                                                           ? sequence
                                                           : projection_chunk_size;
            std::vector<ggml_tensor*> attention_chunks;
            attention_chunks.reserve(static_cast<size_t>((sequence + attention_query_chunk_size - 1) /
                                                         attention_query_chunk_size));
            for (int64_t start = 0; start < sequence; start += attention_query_chunk_size) {
                const int64_t end = std::min(start + attention_query_chunk_size, sequence);
                auto q_chunk = ggml_cont(ctx->ggml_ctx,
                                         ggml_ext_slice(ctx->ggml_ctx, q, 1, start, end));
                auto out_chunk = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                        ctx->backend,
                                                        q_chunk,
                                                        k,
                                                        v,
                                                        static_cast<int>(heads),
                                                        nullptr,
                                                        true,
                                                        true,
                                                        1.f / 128.f,
                                                        sparsity,
                                                        true);
                if (out_chunk->type != wire_type) {
                    out_chunk = ggml_cast(ctx->ggml_ctx, out_chunk, wire_type);
                }
                ggml_format_name(out_chunk,
                                 "h3.sequence.attn.output.%lld",
                                 (long long) (start / attention_query_chunk_size));
                attention_chunks.push_back(out_chunk);
            }
            auto out = concat_balanced(ctx->ggml_ctx, std::move(attention_chunks), 1);
            out = ggml_backend_meta_all_to_all(ctx->ggml_ctx,
                                               ggml_cont(ctx->ggml_ctx, out),
                                               GGML_BACKEND_SPLIT_AXIS_0,
                                               GGML_BACKEND_SPLIT_AXIS_1);
            ggml_set_name(out, "h3.sequence.attn.output.all_to_all");
            std::vector<ggml_tensor*> projected_chunks;
            projected_chunks.reserve(static_cast<size_t>((sequence + projection_chunk_size - 1) /
                                                          projection_chunk_size));
            for (int64_t start = 0; start < sequence; start += projection_chunk_size) {
                const int64_t end = std::min(start + projection_chunk_size, sequence);
                auto projected = out_proj->forward(ctx,
                                                   ggml_ext_slice(ctx->ggml_ctx,
                                                                  out,
                                                                  1,
                                                                  start,
                                                                  end));
                ggml_format_name(projected,
                                 "h3.sequence.attn.projected.%lld",
                                 (long long) (start / projection_chunk_size));
                projected_chunks.push_back(projected);
            }
            return concat_balanced(ctx->ggml_ctx, std::move(projected_chunks), 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe = nullptr) {
            if (chunk_size <= 0 || x->ne[1] <= chunk_size || !ctx->flash_attn_enabled) {
                return forward_full(ctx, x, pe);
            }
            auto outputs = forward_chunks(
                ctx,
                x->ne[1],
                x->ne[2] * x->ne[3],
                pe,
                [&](int64_t start, int64_t end) {
                    return ggml_ext_slice(ctx->ggml_ctx, x, 1, start, end);
                });
            auto output = concat_balanced(ctx->ggml_ctx, std::move(outputs), 1);
            ggml_set_name(output, "h3.attn.output");
            return output;
        }
    };

    struct TokenRefinerBlock : public GGMLBlock {
        TokenRefinerBlock(const Config& config) {
            blocks["norm1"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]  = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps,
                                                         config.attention_chunk_size);
            blocks["mlp"]   = std::make_shared<MLP>(config.hidden_size,
                                                      config.ffn_hidden_size,
                                                      config.mlp_chunk_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            x          = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, norm1->forward(ctx, x)));
            return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x)));
        }
    };

    struct TokenRefiner : public GGMLBlock {
        int64_t num_layers;

        explicit TokenRefiner(const Config& config)
            : num_layers(config.token_refiner_num_layers) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TokenRefinerBlock>(config);
            }
            blocks["final_norm"] = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool cut_after_last = true) {
            auto final_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm"]);
            for (int64_t i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TokenRefinerBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
                const bool is_last = i + 1 == num_layers;
                if (is_last) {
                    x = final_norm->forward(ctx, x);
                }
                if (!is_last || cut_after_last) {
                    sd::ggml_graph_cut::mark_graph_cut(x,
                                                       "minimax_h3.token_refiner.blocks." + std::to_string(i),
                                                       "hidden_states");
                }
            }
            return num_layers == 0 ? final_norm->forward(ctx, x) : x;
        }
    };

    struct AdaLayerNormModulation : public GGMLBlock {
        int64_t hidden_size;
        int expand;
        int modalities;
        bool apply_silu;

        AdaLayerNormModulation(int64_t time_dim,
                               int64_t hidden_size,
                               int expand,
                               int modalities,
                               bool apply_silu,
                               bool force_f32)
            : hidden_size(hidden_size),
              expand(expand),
              modalities(modalities),
              apply_silu(apply_silu) {
            blocks["linear"] = std::make_shared<Linear>(time_dim,
                                                        hidden_size * expand * modalities,
                                                        true,
                                                        force_f32);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            if (apply_silu) {
                t_emb = ggml_silu(ctx->ggml_ctx, t_emb);
            }
            return std::dynamic_pointer_cast<Linear>(blocks["linear"])->forward(ctx, t_emb);
        }
    };

    struct TokenModulationSpan {
        int64_t start;
        int64_t end;
        int64_t modulation_row;
    };

    enum class SequenceKind {
        TEXT,
        CONDITION_VIDEO,
        CONDITION_AUDIO,
        TARGET_AUDIO,
        TARGET_VIDEO,
    };

    struct SequenceSegment {
        int64_t start;
        int64_t end;
        SequenceKind kind;
        int32_t source_index = -1;
    };

    static std::vector<ggml_tensor*> modulation_row(ggml_context* ctx,
                                                    ggml_tensor* projection,
                                                    int64_t hidden_size,
                                                    int expand,
                                                    int modalities,
                                                    int64_t row) {
        int64_t timestep_rows = projection->ne[1];
        auto reshaped         = ggml_reshape_2d(ctx,
                                                projection,
                                                hidden_size * expand,
                                                timestep_rows * modalities);
        auto selected         = ggml_ext_slice(ctx, reshaped, 1, row, row + 1, false);
        GGML_ASSERT(ggml_is_contiguous(selected));
        auto chunks = ggml_ext_chunk(ctx, selected, expand, 0, false);
        for (const auto * chunk : chunks) {
            GGML_ASSERT(ggml_is_contiguous(chunk));
        }
        return chunks;
    }

    static std::vector<ggml_tensor*> sequence_modulation(ggml_context* ctx,
                                                         ggml_tensor* projection,
                                                         ggml_tensor* indices,
                                                         int64_t hidden_size,
                                                         int expand,
                                                         int modalities) {
        const int64_t rows = projection->ne[1] * modalities;
        GGML_ASSERT(ggml_is_contiguous(projection));
        std::vector<ggml_tensor*> result;
        result.reserve(expand);
        for (int i = 0; i < expand; ++i) {
            auto table = ggml_view_2d(ctx,
                                      projection,
                                      hidden_size,
                                      rows,
                                      hidden_size * expand * projection->nb[0],
                                      i * hidden_size * projection->nb[0]);
            auto values = ggml_get_rows(ctx, table, indices);
            ggml_format_name(values, "h3.sequence.modulation.%d", i);
            result.push_back(values);
        }
        return result;
    }

    static ggml_tensor* modulate_segments(ggml_context* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* projection,
                                          const std::vector<TokenModulationSpan>& segments,
                                          int64_t hidden_size,
                                          int expand,
                                          int modalities,
                                          int shift_index,
                                          int scale_index) {
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx,
                                       projection,
                                       hidden_size,
                                       expand,
                                       modalities,
                                       segment.modulation_row);
            auto input = ggml_ext_slice(ctx, x, 1, segment.start, segment.end);
            auto part  = ggml_mul(ctx, input, mods[scale_index]);
            part       = ggml_add_inplace(ctx, part, input);
            part       = ggml_add_inplace(ctx, part, mods[shift_index]);
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    static ggml_tensor* gated_residual_segments(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* update,
                                                ggml_tensor* projection,
                                                const std::vector<TokenModulationSpan>& segments,
                                                int64_t hidden_size,
                                                int gate_index) {
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx, projection, hidden_size, 6, 3, segment.modulation_row);
            auto base = ggml_ext_slice(ctx, x, 1, segment.start, segment.end);
            auto add  = ggml_ext_slice(ctx, update, 1, segment.start, segment.end);
            auto part = ggml_mul(ctx, add, mods[gate_index]);
            part      = ggml_add_inplace(ctx, part, base);
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    static ggml_tensor* normalized_modulated_range(
        GGMLRunnerContext* ctx,
        const std::shared_ptr<RMSNorm>& norm,
        ggml_tensor* x,
        int64_t x_start,
        int64_t start,
        int64_t end,
        ggml_tensor* projection,
        const std::vector<TokenModulationSpan>& segments,
        int64_t hidden_size,
        int shift_index,
        int scale_index) {
        std::vector<ggml_tensor*> outputs;
        for (const auto& segment : segments) {
            int64_t part_start = std::max(start, segment.start);
            int64_t part_end   = std::min(end, segment.end);
            if (part_start >= part_end) {
                continue;
            }
            auto input = ggml_ext_slice(ctx->ggml_ctx,
                                        x,
                                        1,
                                        part_start - x_start,
                                        part_end - x_start,
                                        false);
            if (input->type != GGML_TYPE_F32) {
                input = ggml_cast(ctx->ggml_ctx, input, GGML_TYPE_F32);
            }
            input     = norm->forward(ctx, input);
            auto mods = modulation_row(ctx->ggml_ctx,
                                       projection,
                                       hidden_size,
                                       6,
                                       3,
                                       segment.modulation_row);
            auto output = ggml_mul(ctx->ggml_ctx, input, mods[scale_index]);
            output      = ggml_add_inplace(ctx->ggml_ctx, output, input);
            output      = ggml_add_inplace(ctx->ggml_ctx, output, mods[shift_index]);
            outputs.push_back(output);
        }
        GGML_ASSERT(!outputs.empty());
        return outputs.size() == 1 ? outputs.front()
                                   : concat_balanced(ctx->ggml_ctx, std::move(outputs), 1);
    }

    static ggml_tensor* gated_residual_range(
        GGMLRunnerContext* ctx,
        ggml_tensor* x,
        int64_t x_start,
        ggml_tensor* update,
        int64_t start,
        int64_t end,
        ggml_tensor* projection,
        const std::vector<TokenModulationSpan>& segments,
        int64_t hidden_size,
        int gate_index) {
        std::vector<ggml_tensor*> outputs;
        for (const auto& segment : segments) {
            int64_t part_start = std::max(start, segment.start);
            int64_t part_end   = std::min(end, segment.end);
            if (part_start >= part_end) {
                continue;
            }
            auto base = ggml_ext_slice(ctx->ggml_ctx,
                                       x,
                                       1,
                                       part_start - x_start,
                                       part_end - x_start,
                                       false);
            if (base->type != GGML_TYPE_F32) {
                base = ggml_cast(ctx->ggml_ctx, base, GGML_TYPE_F32);
            }
            auto add = ggml_ext_slice(ctx->ggml_ctx,
                                      update,
                                      1,
                                      part_start - start,
                                      part_end - start,
                                      false);
            auto mods = modulation_row(ctx->ggml_ctx,
                                       projection,
                                       hidden_size,
                                       6,
                                       3,
                                       segment.modulation_row);
            auto output = ggml_mul(ctx->ggml_ctx, add, mods[gate_index]);
            output      = ggml_add_inplace(ctx->ggml_ctx, output, base);
            outputs.push_back(output);
        }
        GGML_ASSERT(!outputs.empty());
        return outputs.size() == 1 ? outputs.front()
                                   : concat_balanced(ctx->ggml_ctx, std::move(outputs), 1);
    }

    struct TransformerBlock : public GGMLBlock {
        Config config;

        explicit TransformerBlock(const Config& config)
            : config(config) {
            blocks["norm1"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]       = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps,
                                                         config.attention_chunk_size,
                                                         config.attention_sparsity);
            blocks["mlp"]        = std::make_shared<MLP>(config.hidden_size,
                                                           config.ffn_hidden_size,
                                                           config.mlp_chunk_size);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            6,
                                                                            3,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
        }

        void set_attention_sparsity(float value) {
            config.attention_sparsity = value;
            std::dynamic_pointer_cast<Attention>(blocks["attn"])->set_sparsity(value);
        }

        ggml_tensor* project_adaln(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            return std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"])
                ->forward(ctx, t_emb);
        }

        std::pair<ggml_tensor*, ggml_tensor*> fill_block_stream_kv_chunk(
            GGMLRunnerContext* ctx,
            ggml_tensor* x,
            int64_t token_start,
            ggml_tensor* mods,
            const std::vector<TokenModulationSpan>& segments,
            ggml_tensor* pe,
            ggml_tensor* key_cache,
            ggml_tensor* value_cache,
            ggml_tensor* cache_rows) {
            const int64_t token_end = token_start + x->ne[1];
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto h = normalized_modulated_range(ctx,
                                                norm1,
                                                x,
                                                token_start,
                                                token_start,
                                                token_end,
                                                mods,
                                                segments,
                                                config.hidden_size,
                                                0,
                                                1);
            return attn->update_kv_cache_chunk(ctx,
                                               h,
                                               pe,
                                               token_start,
                                               key_cache,
                                               value_cache,
                                               cache_rows);
        }

        ggml_tensor* forward_block_stream_chunk(
            GGMLRunnerContext* ctx,
            ggml_tensor* x,
            int64_t token_start,
            ggml_tensor* mods,
            const std::vector<TokenModulationSpan>& segments,
            ggml_tensor* pe,
            ggml_tensor* key_cache,
            ggml_tensor* value_cache,
            ggml_tensor* cache_rows,
            int64_t valid_kv_tokens,
            bool update_kv_cache = true) {
            const int64_t token_end = token_start + x->ne[1];
            auto norm1              = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2              = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn               = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp                = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);

            auto h = normalized_modulated_range(ctx,
                                                norm1,
                                                x,
                                                token_start,
                                                token_start,
                                                token_end,
                                                mods,
                                                segments,
                                                config.hidden_size,
                                                0,
                                                1);
            auto attention_output = update_kv_cache
                                        ? attn->forward_cached_chunk(ctx,
                                                                    h,
                                                                    pe,
                                                                    token_start,
                                                                    key_cache,
                                                                    value_cache,
                                                                    cache_rows,
                                                                    valid_kv_tokens)
                                        : attn->forward_cached_query_chunk(ctx,
                                                                          h,
                                                                          pe,
                                                                          token_start,
                                                                          key_cache,
                                                                          value_cache,
                                                                          valid_kv_tokens);
            auto residual = gated_residual_range(ctx,
                                                 x,
                                                 token_start,
                                                 attention_output,
                                                 token_start,
                                                 token_end,
                                                 mods,
                                                 segments,
                                                 config.hidden_size,
                                                 2);
            if (config.activation_storage_type != GGML_TYPE_F32) {
                residual = ggml_cast(ctx->ggml_ctx,
                                     residual,
                                     config.activation_storage_type);
            }
            h = normalized_modulated_range(ctx,
                                           norm2,
                                           residual,
                                           token_start,
                                           token_start,
                                           token_end,
                                           mods,
                                           segments,
                                           config.hidden_size,
                                           3,
                                           4);
            auto output = gated_residual_range(ctx,
                                               residual,
                                               token_start,
                                               mlp->forward_chunk(ctx, h),
                                               token_start,
                                               token_end,
                                               mods,
                                               segments,
                                               config.hidden_size,
                                               5);
            if (config.activation_storage_type != GGML_TYPE_F32) {
                output = ggml_cast(ctx->ggml_ctx,
                                   output,
                                   config.activation_storage_type);
            }
            return output;
        }

        ggml_tensor* forward_chunked(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* t_emb,
                                     const std::vector<TokenModulationSpan>& segments,
                                     ggml_tensor* pe) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaln = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto mods  = adaln->forward(ctx, t_emb);

            int64_t sequence = x->ne[1];
            auto attention_outputs = attn->forward_chunks(
                ctx,
                sequence,
                x->ne[2] * x->ne[3],
                pe,
                [&](int64_t start, int64_t end) {
                    return normalized_modulated_range(ctx,
                                                      norm1,
                                                      x,
                                                      0,
                                                      start,
                                                      end,
                                                      mods,
                                                      segments,
                                                      config.hidden_size,
                                                      0,
                                                      1);
                });

            std::vector<ggml_tensor*> outputs;
            outputs.reserve(attention_outputs.size());
            size_t chunk_index = 0;
            for (int64_t start = 0; start < sequence; start += attn->chunk_size, ++chunk_index) {
                int64_t end = std::min(start + attn->chunk_size, sequence);
                auto residual = gated_residual_range(ctx,
                                                     x,
                                                     0,
                                                     attention_outputs[chunk_index],
                                                     start,
                                                     end,
                                                     mods,
                                                     segments,
                                                     config.hidden_size,
                                                     2);
                if (config.activation_storage_type != GGML_TYPE_F32) {
                    residual = ggml_cast(ctx->ggml_ctx,
                                         residual,
                                         config.activation_storage_type);
                }
                auto h = normalized_modulated_range(ctx,
                                                    norm2,
                                                    residual,
                                                    start,
                                                    start,
                                                    end,
                                                    mods,
                                                    segments,
                                                    config.hidden_size,
                                                    3,
                                                    4);
                auto output = gated_residual_range(ctx,
                                                   residual,
                                                   start,
                                                   mlp->forward(ctx, h),
                                                   start,
                                                   end,
                                                   mods,
                                                   segments,
                                                   config.hidden_size,
                                                   5);
                if (config.activation_storage_type != GGML_TYPE_F32) {
                    output = ggml_cast(ctx->ggml_ctx,
                                       output,
                                       config.activation_storage_type);
                }
                ggml_format_name(output, "h3.block.output.%zu", chunk_index);
                outputs.push_back(output);
            }
            auto output = concat_balanced(ctx->ggml_ctx, std::move(outputs), 1);
            ggml_set_name(output, "h3.block.output");
            return output;
        }

        ggml_tensor* forward_sequence_parallel(GGMLRunnerContext* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* t_emb,
                                               const std::vector<TokenModulationSpan>& segments,
                                               ggml_tensor* modulation_indices,
                                               ggml_tensor* pe,
                                               const std::string& attention_cut_group) {
            GGML_ASSERT(modulation_indices != nullptr);
            ggml_tensor* residual_input = x;
            if (x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaln = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto projection = adaln->forward(ctx, t_emb);
            const std::string qkv_cut_group = attention_cut_group.empty()
                                                  ? std::string()
                                                  : attention_cut_group + ".qkv";
            if (!qkv_cut_group.empty()) {
                sd::ggml_graph_cut::mark_graph_cut(projection,
                                                   qkv_cut_group,
                                                   "adaln_projection");
                residual_input = ggml_dup(ctx->ggml_ctx, residual_input);
                ggml_set_name(residual_input, "h3.sequence.block.residual_input");
                sd::ggml_graph_cut::mark_graph_cut(residual_input,
                                                   qkv_cut_group,
                                                   "residual_input");
            }
            auto mods = sequence_modulation(ctx->ggml_ctx,
                                            projection,
                                            modulation_indices,
                                            config.hidden_size,
                                            6,
                                            3);
            auto cast_like = [&](ggml_tensor* value, const ggml_tensor* reference) {
                return value->type == reference->type
                           ? value
                           : ggml_cast(ctx->ggml_ctx, value, reference->type);
            };

            auto normalized = norm1->forward_out_of_place(ctx, x);
            auto shift1 = cast_like(mods[0], normalized);
            auto scale1 = cast_like(mods[1], normalized);
            auto h = ggml_add(ctx->ggml_ctx,
                              shift1,
                              ggml_add(ctx->ggml_ctx,
                                       ggml_mul(ctx->ggml_ctx, scale1, normalized),
                                       normalized));
            ggml_set_name(h, "h3.sequence.block.modulated1");

            auto attn_output = attn->forward_sequence_parallel(ctx,
                                                               h,
                                                               pe,
                                                               config.activation_storage_type,
                                                               qkv_cut_group);
            auto gate1 = cast_like(mods[2], attn_output);
            if (residual_input->type != GGML_TYPE_F32) {
                residual_input = ggml_cast(ctx->ggml_ctx,
                                           residual_input,
                                           GGML_TYPE_F32);
            }
            auto residual = ggml_add(ctx->ggml_ctx,
                                     residual_input,
                                     ggml_mul(ctx->ggml_ctx, attn_output, gate1));
            if (config.activation_storage_type != GGML_TYPE_F32) {
                residual = ggml_cast(ctx->ggml_ctx,
                                     residual,
                                     config.activation_storage_type);
            }
            ggml_set_name(residual, "h3.sequence.block.residual1");
            if (!attention_cut_group.empty()) {
                sd::ggml_graph_cut::mark_graph_cut(residual,
                                                   attention_cut_group,
                                                   "hidden_states");
            }

            auto residual_for_mlp = residual;
            if (residual_for_mlp->type != GGML_TYPE_F32) {
                residual_for_mlp = ggml_cast(ctx->ggml_ctx,
                                             residual_for_mlp,
                                             GGML_TYPE_F32);
            }
            normalized = norm2->forward_out_of_place(ctx, residual_for_mlp);
            auto shift2 = cast_like(mods[3], normalized);
            auto scale2 = cast_like(mods[4], normalized);
            h = ggml_add(ctx->ggml_ctx,
                         shift2,
                         ggml_add(ctx->ggml_ctx,
                                  ggml_mul(ctx->ggml_ctx, scale2, normalized),
                                  normalized));
            ggml_set_name(h, "h3.sequence.block.modulated2");

            auto mlp_output = mlp->forward(ctx, h);
            auto gate2 = cast_like(mods[5], mlp_output);
            auto output = ggml_add(ctx->ggml_ctx,
                                   residual_for_mlp,
                                   ggml_mul(ctx->ggml_ctx, mlp_output, gate2));
            if (config.activation_storage_type != GGML_TYPE_F32) {
                output = ggml_cast(ctx->ggml_ctx, output, config.activation_storage_type);
            }
            ggml_set_name(output, "h3.sequence.block.output");
            return output;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_emb,
                             const std::vector<TokenModulationSpan>& segments,
                             ggml_tensor* pe,
                             ggml_tensor* modulation_indices = nullptr,
                             const std::string& attention_cut_group = "") {
            if (config.sequence_parallel) {
                return forward_sequence_parallel(ctx,
                                                 x,
                                                 t_emb,
                                                 segments,
                                                 modulation_indices,
                                                 pe,
                                                 attention_cut_group);
            }
            auto attn = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            if (attn->chunk_size > 0 && x->ne[1] > attn->chunk_size && ctx->flash_attn_enabled) {
                return forward_chunked(ctx, x, t_emb, segments, pe);
            }
            if (x->type != GGML_TYPE_F32) {
                x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
            }
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaln = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto mods  = adaln->forward(ctx, t_emb);

            auto normalized = norm1->forward(ctx, x);
            ggml_set_name(normalized, "h3.block.norm1");
            auto h = modulate_segments(ctx->ggml_ctx,
                                       normalized,
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       0,
                                       1);
            ggml_set_name(h, "h3.block.modulated1");
            x      = gated_residual_segments(ctx->ggml_ctx,
                                             x,
                                             attn->forward(ctx, h, pe),
                                             mods,
                                             segments,
                                             config.hidden_size,
                                             2);
            ggml_set_name(x, "h3.block.residual1");
            normalized = norm2->forward(ctx, x);
            ggml_set_name(normalized, "h3.block.norm2");
            h      = modulate_segments(ctx->ggml_ctx,
                                       normalized,
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       3,
                                       4);
            ggml_set_name(h, "h3.block.modulated2");
            auto output = gated_residual_segments(ctx->ggml_ctx,
                                                  x,
                                                  mlp->forward(ctx, h),
                                                  mods,
                                                  segments,
                                                  config.hidden_size,
                                                  5);
            if (config.activation_storage_type != GGML_TYPE_F32) {
                output = ggml_cast(ctx->ggml_ctx,
                                   output,
                                   config.activation_storage_type);
            }
            ggml_set_name(output, "h3.block.output");
            return output;
        }
    };

    struct FinalLayer : public GGMLBlock {
        Config config;

        explicit FinalLayer(const Config& config)
            : config(config) {
            int64_t video_dim    = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["norm"]       = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            2,
                                                                            1,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
            blocks["video_out"]  = std::make_shared<Linear>(config.hidden_size, video_dim, true, true);
            blocks["audio_out"]  = std::make_shared<Linear>(config.hidden_size, config.audio_latent_channels, true, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* t_emb,
                                                      const TokenModulationSpan& video,
                                                      const TokenModulationSpan& audio) {
            auto norm      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto adaln     = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto video_out = std::dynamic_pointer_cast<Linear>(blocks["video_out"]);
            auto audio_out = std::dynamic_pointer_cast<Linear>(blocks["audio_out"]);
            auto mods      = adaln->forward(ctx, t_emb);
            auto apply     = [&](const TokenModulationSpan& segment) {
                auto row   = modulation_row(ctx->ggml_ctx, mods, config.hidden_size, 2, 1, segment.modulation_row);
                auto value = norm->forward(ctx, ggml_ext_slice(ctx->ggml_ctx, x, 1, segment.start, segment.end));
                return ggml_add(ctx->ggml_ctx,
                                    ggml_add(ctx->ggml_ctx, value, ggml_mul(ctx->ggml_ctx, value, row[1])),
                                    row[0]);
            };
            return {video_out->forward(ctx, apply(video)),
                    audio_out->forward(ctx, apply(audio))};
        }
    };

    struct MiniMaxH3Transformer3DModel : public GGMLBlock {
        Config config;

        explicit MiniMaxH3Transformer3DModel(const Config& config)
            : config(config) {
            int64_t video_dim          = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["video_patch_proj"] = std::make_shared<Linear>(video_dim, config.hidden_size, true, true);
            blocks["audio_patch_proj"] = std::make_shared<Linear>(config.audio_latent_channels, config.hidden_size, true, true);
            blocks["condition_proj"]   = std::make_shared<Linear>(config.text_dim, config.hidden_size, true);
            if (!config.uses_adaln_curves()) {
                blocks["time_embedder"] = std::make_shared<TimeEmbedder>(config.timestep_input_dim,
                                                                         config.time_embed_hidden_size,
                                                                         config.time_embed_dim);
            }
            blocks["token_refiner"] = std::make_shared<TokenRefiner>(config);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(config);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(config);
        }

        void set_attention_sparsity(float value) {
            config.attention_sparsity = value;
            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                block->set_attention_sparsity(value);
            }
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensors = {},
                         const std::string prefix            = "") override {
            GGMLBlock::init_params(ctx, tensors, prefix);
            params["rope.inv_freq"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.rope_inv_freq_len);
            if (config.uses_adaln_curves()) {
                params["adaln_t_table"] = ggml_new_tensor_2d(ctx,
                                                             GGML_TYPE_F32,
                                                             config.time_embed_dim,
                                                             config.adaln_curve_grid);
            }
        }

        ggml_tensor* refine_context(GGMLRunnerContext* ctx,
                                    ggml_tensor* context,
                                    bool cut_after_last_refiner = true) {
            if (context->ne[0] == config.hidden_size) {
                return context;
            }
            GGML_ASSERT(context->ne[0] == config.text_dim);
            auto condition_proj = std::dynamic_pointer_cast<Linear>(blocks["condition_proj"]);
            auto token_refiner  = std::dynamic_pointer_cast<TokenRefiner>(blocks["token_refiner"]);
            auto projected      = condition_proj->forward(ctx, context);
            sd::ggml_graph_cut::mark_graph_cut(projected,
                                               "minimax_h3.condition_proj",
                                               "hidden_states");
            return token_refiner->forward(ctx, projected, cut_after_last_refiner);
        }

        ggml_tensor* time_embedding(GGMLRunnerContext* ctx,
                                    ggml_tensor* timestep_features,
                                    ggml_tensor* curve_indices,
                                    ggml_tensor* curve_upper_indices,
                                    ggml_tensor* curve_fractions) {
            if (!config.uses_adaln_curves()) {
                return std::dynamic_pointer_cast<TimeEmbedder>(blocks["time_embedder"])->forward(ctx, timestep_features);
            }
            auto lower = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_indices);
            auto upper = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_upper_indices);
            return ggml_add(ctx->ggml_ctx,
                            lower,
                            ggml_mul(ctx->ggml_ctx,
                                     ggml_sub(ctx->ggml_ctx, upper, lower),
                                     curve_fractions));
        }

        ggml_tensor* build_rope(GGMLRunnerContext* ctx,
                                ggml_tensor* position_ids) {
            auto inv            = ggml_reshape_2d(ctx->ggml_ctx,
                                                  params["rope.inv_freq"],
                                                  config.rope_inv_freq_len,
                                                  1);
            ggml_tensor* angles = nullptr;
            for (int axis = 0; axis < 3; ++axis) {
                auto pos          = ggml_ext_slice(ctx->ggml_ctx, position_ids, 0, axis, axis + 1);
                auto expanded_inv = ggml_repeat_4d(ctx->ggml_ctx,
                                                   inv,
                                                   inv->ne[0],
                                                   pos->ne[1],
                                                   1,
                                                   1);
                auto a            = ggml_mul(ctx->ggml_ctx, expanded_inv, pos);
                angles            = angles == nullptr ? a : ggml_concat(ctx->ggml_ctx, angles, a, 0);
            }
            auto c  = ggml_reshape_4d(ctx->ggml_ctx, ggml_cos(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto s  = ggml_reshape_4d(ctx->ggml_ctx, ggml_sin(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto ns = ggml_neg(ctx->ggml_ctx, s);
            auto pe = ggml_concat(ctx->ggml_ctx, c, ns, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, s, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, c, 0);
            return ggml_reshape_4d(ctx->ggml_ctx, pe, 2, 2, angles->ne[0], angles->ne[1]);
        }

        struct PreparedBlockInputs {
            ggml_tensor* hidden = nullptr;
            ggml_tensor* timestep_embedding = nullptr;
            ggml_tensor* rotary_embedding = nullptr;
        };

        PreparedBlockInputs prepare_block_inputs(
            GGMLRunnerContext* ctx,
            ggml_tensor* video,
            ggml_tensor* audio,
            ggml_tensor* context,
            const std::vector<ggml_tensor*>& condition_videos,
            const std::vector<ggml_tensor*>& condition_audios,
            ggml_tensor* position_ids,
            ggml_tensor* timestep_features,
            ggml_tensor* curve_indices,
            ggml_tensor* curve_upper_indices,
            ggml_tensor* curve_fractions,
            const std::vector<SequenceSegment>& sequence_segments) {
            auto video_proj = std::dynamic_pointer_cast<Linear>(blocks["video_patch_proj"]);
            auto audio_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patch_proj"]);

            std::vector<std::pair<int64_t, int64_t>> condition_video_ranges;
            ggml_tensor* video_rows = nullptr;
            int64_t video_offset    = 0;
            for (auto condition : condition_videos) {
                auto rows = DiT::patchify_3d(ctx->ggml_ctx,
                                             condition,
                                             config.patch_t,
                                             config.patch_h,
                                             config.patch_w,
                                             1,
                                             true);
                condition_video_ranges.push_back({video_offset, video_offset + rows->ne[1]});
                video_offset += rows->ne[1];
                video_rows = video_rows == nullptr ? rows
                                                   : ggml_concat(ctx->ggml_ctx, video_rows, rows, 1);
            }
            auto target_video_rows = DiT::patchify_3d(ctx->ggml_ctx,
                                                      video,
                                                      config.patch_t,
                                                      config.patch_h,
                                                      config.patch_w,
                                                      1,
                                                      true);
            const std::pair<int64_t, int64_t> target_video_range = {
                video_offset, video_offset + target_video_rows->ne[1]};
            video_rows = video_rows == nullptr ? target_video_rows
                                               : ggml_concat(ctx->ggml_ctx,
                                                             video_rows,
                                                             target_video_rows,
                                                             1);
            auto video_embeds = video_proj->forward(ctx, video_rows);

            auto pack_audio_rows = [&](ggml_tensor* value) {
                value = ggml_cont(ctx->ggml_ctx,
                                  ggml_ext_torch_permute(ctx->ggml_ctx, value, 2, 0, 1, 3));
                return ggml_reshape_3d(ctx->ggml_ctx,
                                       value,
                                       value->ne[0],
                                       value->ne[1] * value->ne[2],
                                       value->ne[3]);
            };
            std::vector<std::pair<int64_t, int64_t>> condition_audio_ranges;
            ggml_tensor* audio_rows = nullptr;
            int64_t audio_offset    = 0;
            for (auto condition : condition_audios) {
                auto rows = pack_audio_rows(condition);
                condition_audio_ranges.push_back({audio_offset, audio_offset + rows->ne[1]});
                audio_offset += rows->ne[1];
                audio_rows = audio_rows == nullptr ? rows
                                                   : ggml_concat(ctx->ggml_ctx, audio_rows, rows, 1);
            }
            audio = pack_audio_rows(audio);
            const std::pair<int64_t, int64_t> target_audio_range = {
                audio_offset, audio_offset + audio->ne[1]};
            audio_rows = audio_rows == nullptr ? audio
                                               : ggml_concat(ctx->ggml_ctx, audio_rows, audio, 1);
            auto audio_embeds = audio_proj->forward(ctx, audio_rows);

            ggml_tensor* h = nullptr;
            auto append = [&](ggml_tensor* value) {
                h = h == nullptr ? value : ggml_concat(ctx->ggml_ctx, h, value, 1);
            };
            for (const auto& sequence : sequence_segments) {
                if (sequence.kind == SequenceKind::TEXT) {
                    append(context);
                } else if (sequence.kind == SequenceKind::CONDITION_VIDEO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_video_ranges.size()));
                    const auto range = condition_video_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, video_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::CONDITION_AUDIO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_audio_ranges.size()));
                    const auto range = condition_audio_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, audio_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::TARGET_AUDIO) {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          audio_embeds,
                                          1,
                                          target_audio_range.first,
                                          target_audio_range.second));
                } else {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          video_embeds,
                                          1,
                                          target_video_range.first,
                                          target_video_range.second));
                }
            }
            GGML_ASSERT(h != nullptr);
            auto t_emb = time_embedding(ctx,
                                        timestep_features,
                                        curve_indices,
                                        curve_upper_indices,
                                        curve_fractions);
            auto pe = build_rope(ctx, position_ids);
            if (config.activation_storage_type != GGML_TYPE_F32) {
                h = ggml_cast(ctx->ggml_ctx, h, config.activation_storage_type);
            }
            return {h, t_emb, pe};
        }

        std::shared_ptr<TransformerBlock> transformer_block(int64_t index) const {
            if (index < 0 || index >= config.execution_layers) {
                return nullptr;
            }
            const auto it = blocks.find("blocks." + std::to_string(index));
            return it == blocks.end()
                       ? nullptr
                       : std::dynamic_pointer_cast<TransformerBlock>(it->second);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward_final(
            GGMLRunnerContext* ctx,
            ggml_tensor* h,
            ggml_tensor* t_emb,
            ggml_tensor* video,
            ggml_tensor* audio,
            const TokenModulationSpan& video_segment,
            const TokenModulationSpan& audio_segment,
            float audio_slope) {
            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            if (h->type != GGML_TYPE_F32) {
                h = ggml_cast(ctx->ggml_ctx, h, GGML_TYPE_F32);
            }
            auto output    = final_layer->forward(ctx, h, t_emb, video_segment, audio_segment);
            auto video_out = DiT::unpatchify_3d(ctx->ggml_ctx,
                                                output.first,
                                                video->ne[2] / config.patch_t,
                                                video->ne[1] / config.patch_h,
                                                video->ne[0] / config.patch_w,
                                                config.patch_t,
                                                config.patch_h,
                                                config.patch_w,
                                                true);
            auto audio_out = ggml_reshape_4d(ctx->ggml_ctx,
                                             output.second,
                                             config.audio_latent_channels,
                                             audio->ne[1] / 2,
                                             2,
                                             audio->ne[2]);
            audio_out = ggml_cont(ctx->ggml_ctx,
                                  ggml_ext_torch_permute(ctx->ggml_ctx, audio_out, 1, 2, 0, 3));
            video_out = ggml_ext_scale(ctx->ggml_ctx, video_out, -1.f);
            audio_out = ggml_ext_scale(ctx->ggml_ctx, audio_out, -audio_slope);
            return {video_out, audio_out};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* video,
                                                      ggml_tensor* audio,
                                                      ggml_tensor* context,
                                                      const std::vector<ggml_tensor*>& condition_videos,
                                                      const std::vector<ggml_tensor*>& condition_audios,
                                                      ggml_tensor* position_ids,
                                                      ggml_tensor* modulation_indices,
                                                      ggml_tensor* timestep_features,
                                                      ggml_tensor* curve_indices,
                                                      ggml_tensor* curve_upper_indices,
                                                      ggml_tensor* curve_fractions,
                                                      const std::vector<TokenModulationSpan>& segments,
                                                      const std::vector<SequenceSegment>& sequence_segments,
                                                      const TokenModulationSpan& video_segment,
                                                      const TokenModulationSpan& audio_segment,
                                                      float audio_slope) {
            auto video_proj = std::dynamic_pointer_cast<Linear>(blocks["video_patch_proj"]);
            auto audio_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patch_proj"]);

            std::vector<std::pair<int64_t, int64_t>> condition_video_ranges;
            ggml_tensor* video_rows = nullptr;
            int64_t video_offset    = 0;
            for (auto condition : condition_videos) {
                auto rows = DiT::patchify_3d(ctx->ggml_ctx,
                                             condition,
                                             config.patch_t,
                                             config.patch_h,
                                             config.patch_w,
                                             1,
                                             true);
                condition_video_ranges.push_back({video_offset, video_offset + rows->ne[1]});
                video_offset += rows->ne[1];
                video_rows = video_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, video_rows, rows, 1);
            }
            auto target_video_rows                         = DiT::patchify_3d(ctx->ggml_ctx,
                                                                              video,
                                                                              config.patch_t,
                                                                              config.patch_h,
                                                                              config.patch_w,
                                                                              1,
                                                                              true);
            std::pair<int64_t, int64_t> target_video_range = {video_offset, video_offset + target_video_rows->ne[1]};
            video_rows                                     = video_rows == nullptr ? target_video_rows
                                                                                   : ggml_concat(ctx->ggml_ctx, video_rows, target_video_rows, 1);
            auto video_embeds                              = video_proj->forward(ctx, video_rows);

            auto pack_audio_rows = [&](ggml_tensor* value) {
                value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, value, 2, 0, 1, 3));
                return ggml_reshape_3d(ctx->ggml_ctx,
                                       value,
                                       value->ne[0],
                                       value->ne[1] * value->ne[2],
                                       value->ne[3]);
            };
            std::vector<std::pair<int64_t, int64_t>> condition_audio_ranges;
            ggml_tensor* audio_rows = nullptr;
            int64_t audio_offset    = 0;
            for (auto condition : condition_audios) {
                auto rows = pack_audio_rows(condition);
                condition_audio_ranges.push_back({audio_offset, audio_offset + rows->ne[1]});
                audio_offset += rows->ne[1];
                audio_rows = audio_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, audio_rows, rows, 1);
            }
            audio                                          = pack_audio_rows(audio);
            std::pair<int64_t, int64_t> target_audio_range = {audio_offset, audio_offset + audio->ne[1]};
            audio_rows                                     = audio_rows == nullptr ? audio : ggml_concat(ctx->ggml_ctx, audio_rows, audio, 1);
            auto audio_embeds                              = audio_proj->forward(ctx, audio_rows);
            context                                        = refine_context(ctx, context);

            ggml_tensor* h = nullptr;
            auto append    = [&](ggml_tensor* value) {
                h = h == nullptr ? value : ggml_concat(ctx->ggml_ctx, h, value, 1);
            };
            for (const auto& sequence : sequence_segments) {
                if (sequence.kind == SequenceKind::TEXT) {
                    append(context);
                } else if (sequence.kind == SequenceKind::CONDITION_VIDEO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_video_ranges.size()));
                    auto range = condition_video_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, video_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::CONDITION_AUDIO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_audio_ranges.size()));
                    auto range = condition_audio_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, audio_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::TARGET_AUDIO) {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          audio_embeds,
                                          1,
                                          target_audio_range.first,
                                          target_audio_range.second));
                } else {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          video_embeds,
                                          1,
                                          target_video_range.first,
                                          target_video_range.second));
                }
            }
            GGML_ASSERT(h != nullptr);

            auto t_emb = time_embedding(ctx,
                                        timestep_features,
                                        curve_indices,
                                        curve_upper_indices,
                                        curve_fractions);
            auto pe    = build_rope(ctx, position_ids);
            if (config.activation_storage_type != GGML_TYPE_F32) {
                h = ggml_cast(ctx->ggml_ctx, h, config.activation_storage_type);
            }
            if (config.sequence_parallel) {
                GGML_ASSERT(modulation_indices != nullptr);
                h = ggml_backend_meta_scatter(ctx->ggml_ctx,
                                              ggml_cont(ctx->ggml_ctx, h),
                                              GGML_BACKEND_SPLIT_AXIS_1);
                modulation_indices = ggml_backend_meta_scatter(ctx->ggml_ctx,
                                                               modulation_indices,
                                                               GGML_BACKEND_SPLIT_AXIS_0);
                ggml_set_name(h, "h3.sequence.hidden_states");
                ggml_set_name(pe, "h3.sequence.rotary_embeddings");
                ggml_set_name(modulation_indices, "h3.sequence.modulation_indices");
            }
            if (config.activation_storage_type != GGML_TYPE_F32 || config.sequence_parallel) {
                ggml_set_name(h, "h3.block.input");
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "minimax_h3.input_pack",
                                                   "hidden_states");
                if (config.sequence_parallel) {
                    sd::ggml_graph_cut::mark_graph_cut(pe,
                                                       "minimax_h3.input_pack",
                                                       "rotary_embeddings");
                    sd::ggml_graph_cut::mark_graph_cut(modulation_indices,
                                                       "minimax_h3.input_pack",
                                                       "modulation_indices");
                    sd::ggml_graph_cut::mark_graph_cut(t_emb,
                                                       "minimax_h3.input_pack",
                                                       "timestep_embedding");
                }
            }
            for (int64_t i = 0; i < config.execution_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                h          = block->forward(ctx,
                                            h,
                                            t_emb,
                                            segments,
                                            pe,
                                            modulation_indices,
                                            "minimax_h3.blocks." + std::to_string(i) + ".attention");
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "minimax_h3.blocks." + std::to_string(i),
                                                   "hidden_states");
            }

            if (config.sequence_parallel) {
                h = ggml_backend_meta_all_gather(ctx->ggml_ctx,
                                                 ggml_cont(ctx->ggml_ctx, h),
                                                 GGML_BACKEND_SPLIT_AXIS_1);
                ggml_set_name(h, "h3.sequence.hidden_states.gathered");
            }

            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            if (h->type != GGML_TYPE_F32) {
                h = ggml_cast(ctx->ggml_ctx, h, GGML_TYPE_F32);
            }
            auto output      = final_layer->forward(ctx, h, t_emb, video_segment, audio_segment);
            auto video_out   = DiT::unpatchify_3d(ctx->ggml_ctx,
                                                  output.first,
                                                  video->ne[2] / config.patch_t,
                                                  video->ne[1] / config.patch_h,
                                                  video->ne[0] / config.patch_w,
                                                  config.patch_t,
                                                  config.patch_h,
                                                  config.patch_w,
                                                  true);
            auto audio_out   = ggml_reshape_4d(ctx->ggml_ctx,
                                               output.second,
                                               config.audio_latent_channels,
                                               audio->ne[1] / 2,
                                               2,
                                               audio->ne[2]);
            audio_out        = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, audio_out, 1, 2, 0, 3));
            video_out        = ggml_ext_scale(ctx->ggml_ctx, video_out, -1.f);
            audio_out        = ggml_ext_scale(ctx->ggml_ctx, audio_out, -audio_slope);
            return {video_out, audio_out};
        }
    };

    struct PackedSequenceLayout {
        std::vector<float> positions;
        std::vector<TokenModulationSpan> segments;
        std::vector<SequenceSegment> sequence_segments;
        TokenModulationSpan video_segment{};
        TokenModulationSpan audio_segment{};
        std::vector<float> timesteps;
    };

    static float video_span(int64_t frame) {
        static const int spans[5] = {1, 4, 4, 4, 4};
        return FRAME_RESCALE * spans[frame % 5];
    }

    static std::vector<float> spatial_axis(int64_t dim,
                                           float sqrt_area) {
        int64_t count = dim / 2;
        float ratio   = static_cast<float>(dim) / sqrt_area;
        std::vector<float> result(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            result[static_cast<size_t>(i)] =
                (static_cast<float>(i) * (ratio / count) + (1.f - ratio) * 0.5f) * 32.f;
        }
        return result;
    }

    static int find_or_add_timestep(std::vector<float>* values, float value) {
        auto it = std::find(values->begin(), values->end(), value);
        if (it != values->end()) {
            return static_cast<int>(it - values->begin());
        }
        values->push_back(value);
        return static_cast<int>(values->size() - 1);
    }

    static PackedSequenceLayout build_layout(int64_t text_len,
                                             int64_t latent_t,
                                             int64_t latent_h,
                                             int64_t latent_w,
                                             int64_t audio_t,
                                             const std::vector<sd::Tensor<float>>& condition_videos,
                                             const std::vector<sd::Tensor<float>>& condition_audios,
                                             const sd::Tensor<int32_t>& keyframe_indices,
                                             const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                             const sd::Tensor<int32_t>& text_tags,
                                             float video_t,
                                             float audio_timestep,
                                             bool distinct_timestep_rows = false) {
        PackedSequenceLayout layout;
        float sqrt_area    = std::sqrt(static_cast<float>(latent_h * latent_w));
        auto h_axis        = spatial_axis(latent_h, sqrt_area);
        auto w_axis        = spatial_axis(latent_w, sqrt_area);
        int64_t frame_rows = static_cast<int64_t>(h_axis.size() * w_axis.size());
        int64_t row        = 0;

        auto append_position = [&](float t, float h, float w) {
            layout.positions.push_back(t);
            layout.positions.push_back(h);
            layout.positions.push_back(w);
        };
        for (int64_t i = 0; i < text_len; ++i) {
            append_position(static_cast<float>(i), 0.f, 0.f);
        }
        layout.sequence_segments.push_back({0, text_len, SequenceKind::TEXT});

        auto add_timestep = [&](float value) {
            if (distinct_timestep_rows) {
                layout.timesteps.push_back(value);
                return static_cast<int>(layout.timesteps.size() - 1);
            }
            return find_or_add_timestep(&layout.timesteps, value);
        };
        int video_time_row           = add_timestep(video_t);
        int audio_time_row           = add_timestep(audio_timestep);
        int condition_time_row       = add_timestep(std::max(video_t, VISUAL_COND_TIMESTEP));
        int audio_condition_time_row = add_timestep(std::max(audio_timestep, 1.f));

        int64_t run_start = 0;
        int current_tag   = text_tags.empty() ? 1 : text_tags[0];
        for (int64_t i = 1; i <= text_len; ++i) {
            int tag = i < text_len && !text_tags.empty() ? text_tags[i] : -1;
            if (i == text_len || tag != current_tag) {
                layout.segments.push_back({run_start,
                                           i,
                                           video_time_row * 3 + current_tag});
                run_start   = i;
                current_tag = tag;
            }
        }
        row = text_len;

        auto condition_spatial_axes = [&](const sd::Tensor<float>& condition) {
            float area = std::sqrt(static_cast<float>(condition.shape()[0] * condition.shape()[1]));
            return std::make_pair(spatial_axis(condition.shape()[1], area),
                                  spatial_axis(condition.shape()[0], area));
        };
        auto append_video_positions = [&](const sd::Tensor<float>& condition,
                                          float cursor) {
            auto axes = condition_spatial_axes(condition);
            for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                for (float h : axes.first) {
                    for (float w : axes.second) {
                        append_position(cursor, h, w);
                    }
                }
                cursor += video_span(t);
            }
            return cursor;
        };
        auto append_audio_positions = [&](int64_t length,
                                          float cursor,
                                          float w_low,
                                          float w_high) {
            for (int channel = 0; channel < 2; ++channel) {
                float w = channel == 0 ? w_low : w_high;
                for (int64_t t = 0; t < length; ++t) {
                    append_position(cursor + static_cast<float>(t), 0.f, w);
                }
            }
        };

        float cursor = static_cast<float>(text_len);
        if (reference_blocks.empty()) {
            float video_duration = 0.f;
            for (int64_t t = 0; t < latent_t; ++t) {
                video_duration += video_span(t);
            }
            for (size_t index = 0; index < condition_videos.size(); ++index) {
                const auto& condition = condition_videos[index];
                auto axes             = condition_spatial_axes(condition);
                int64_t count         = condition.shape()[2] *
                                static_cast<int64_t>(axes.first.size() * axes.second.size());
                bool is_first    = keyframe_indices.empty() || keyframe_indices[static_cast<int64_t>(index)] == 0;
                float keyframe_t = is_first ? static_cast<float>(text_len)
                                            : static_cast<float>(text_len) + video_duration - FRAME_RESCALE;
                for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                    for (float h : axes.first) {
                        for (float w : axes.second) {
                            append_position(keyframe_t, h, w);
                        }
                    }
                }
                layout.sequence_segments.push_back({row,
                                                    row + count,
                                                    SequenceKind::CONDITION_VIDEO,
                                                    static_cast<int32_t>(index)});
                layout.segments.push_back({row, row + count, condition_time_row * 3});
                row += count;
            }
        } else {
            for (const auto& block : reference_blocks) {
                const sd::Tensor<float>* ref_video = nullptr;
                const sd::Tensor<float>* ref_audio = nullptr;
                if (block.video_index >= 0) {
                    GGML_ASSERT(block.video_index < static_cast<int32_t>(condition_videos.size()));
                    ref_video = &condition_videos[static_cast<size_t>(block.video_index)];
                }
                if (block.audio_index >= 0) {
                    GGML_ASSERT(block.audio_index < static_cast<int32_t>(condition_audios.size()));
                    ref_audio = &condition_audios[static_cast<size_t>(block.audio_index)];
                }

                float block_end = cursor;
                if (block.kind == MiniMaxH3ReferenceKind::AUDIO ||
                    block.kind == MiniMaxH3ReferenceKind::VIDEO_AUDIO) {
                    GGML_ASSERT(ref_audio != nullptr);
                    float w_low  = w_axis.front();
                    float w_high = w_axis.back();
                    if (ref_video != nullptr) {
                        auto axes = condition_spatial_axes(*ref_video);
                        w_low     = axes.second.front();
                        w_high    = axes.second.back();
                    }
                    int64_t count = ref_audio->shape()[0] * 2;
                    append_audio_positions(ref_audio->shape()[0], cursor, w_low, w_high);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_AUDIO,
                                                        block.audio_index});
                    layout.segments.push_back({row,
                                               row + count,
                                               audio_condition_time_row * 3 + 2});
                    row += count;
                    block_end = std::max(block_end, cursor + static_cast<float>(ref_audio->shape()[0]));
                }

                if (block.kind != MiniMaxH3ReferenceKind::AUDIO) {
                    GGML_ASSERT(ref_video != nullptr);
                    auto axes     = condition_spatial_axes(*ref_video);
                    int64_t count = ref_video->shape()[2] *
                                    static_cast<int64_t>(axes.first.size() * axes.second.size());
                    float video_end = append_video_positions(*ref_video, cursor);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_VIDEO,
                                                        block.video_index});
                    layout.segments.push_back({row, row + count, condition_time_row * 3});
                    row += count;
                    block_end = block.kind == MiniMaxH3ReferenceKind::IMAGE
                                    ? std::max(block_end, cursor + 1.f)
                                    : std::max(block_end, video_end);
                }
                cursor = block_end;
            }
        }

        int64_t audio_start = row;
        append_audio_positions(audio_t, cursor, w_axis.front(), w_axis.back());
        layout.audio_segment = {audio_start, row + audio_t * 2, audio_time_row};
        layout.sequence_segments.push_back({audio_start,
                                            row + audio_t * 2,
                                            SequenceKind::TARGET_AUDIO});
        layout.segments.push_back({audio_start,
                                   row + audio_t * 2,
                                   audio_time_row * 3 + 2});
        row += audio_t * 2;

        int64_t video_start = row;
        for (int64_t t = 0; t < latent_t; ++t) {
            for (float h : h_axis) {
                for (float w : w_axis) {
                    append_position(cursor, h, w);
                }
            }
            cursor += video_span(t);
        }
        int64_t video_rows   = latent_t * frame_rows;
        layout.video_segment = {video_start, video_start + video_rows, video_time_row};
        layout.sequence_segments.push_back({video_start,
                                            video_start + video_rows,
                                            SequenceKind::TARGET_VIDEO});
        layout.segments.push_back({video_start,
                                   video_start + video_rows,
                                   video_time_row * 3});
        return layout;
    }

    struct BlockStreamPersistentArena {
        ggml_backend_t backend = nullptr;
        ggml_context* context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;

        ~BlockStreamPersistentArena() {
            reset();
        }

        BlockStreamPersistentArena() = default;
        BlockStreamPersistentArena(const BlockStreamPersistentArena&) = delete;
        BlockStreamPersistentArena& operator=(const BlockStreamPersistentArena&) = delete;

        void reset() {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            if (context != nullptr) {
                ggml_free(context);
                context = nullptr;
            }
            backend = nullptr;
        }

        bool begin(ggml_backend_t target, size_t tensor_capacity) {
            reset();
            backend = target;
            ggml_init_params params;
            params.mem_size   = std::max<size_t>(tensor_capacity, 1) * ggml_tensor_overhead();
            params.mem_buffer = nullptr;
            params.no_alloc   = true;
            context           = ggml_init(params);
            return context != nullptr;
        }

        ggml_tensor* tensor_1d(ggml_type type, int64_t ne0, const char* name) {
            auto tensor = ggml_new_tensor_1d(context, type, ne0);
            ggml_set_name(tensor, name);
            return tensor;
        }

        ggml_tensor* tensor_2d(ggml_type type,
                               int64_t ne0,
                               int64_t ne1,
                               const char* name) {
            auto tensor = ggml_new_tensor_2d(context, type, ne0, ne1);
            ggml_set_name(tensor, name);
            return tensor;
        }

        ggml_tensor* tensor_4d(ggml_type type,
                               int64_t ne0,
                               int64_t ne1,
                               int64_t ne2,
                               int64_t ne3,
                               const char* name) {
            auto tensor = ggml_new_tensor_4d(context, type, ne0, ne1, ne2, ne3);
            ggml_set_name(tensor, name);
            return tensor;
        }

        bool allocate() {
            buffer = ggml_backend_alloc_ctx_tensors(context, backend);
            return buffer != nullptr;
        }
    };

    struct BlockStreamLayerState {
        int owner_device = -1;
        ggml_tensor* key_cache = nullptr;
        ggml_tensor* value_cache = nullptr;
        ggml_tensor* modulation = nullptr;
        bool cache_valid = false;
        int64_t fresh_prefix_end = 0;
    };

    struct BlockStreamDeviceTensors {
        ggml_tensor* hidden_input = nullptr;
        ggml_tensor* final_hidden = nullptr;
        ggml_tensor* timestep_embedding = nullptr;
        ggml_tensor* rotary_embedding = nullptr;
        ggml_tensor* cache_rows = nullptr;
        std::vector<ggml_tensor*> full_sequence_buffers;
        std::vector<ggml_tensor*> send_buffers;
        std::vector<ggml_tensor*> receive_buffers;
        std::vector<ggml_tensor*> send_last_buffers;
        std::vector<ggml_tensor*> receive_last_buffers;
    };

    struct BlockStreamGenerationState {
        int64_t sequence = 0;
        int64_t batch = 0;
        int64_t timestep_rows = 0;
        int64_t chunk_size = 0;
        int chunk_count = 0;
        int final_owner = 0;
        int64_t evaluation_count = 0;
        const void* conditioning_identity = nullptr;
        std::vector<int64_t> packed_shape;
        std::vector<int64_t> video_shape;
        std::vector<int64_t> audio_shape;
        std::vector<std::unique_ptr<BlockStreamPersistentArena>> arenas;
        std::vector<BlockStreamDeviceTensors> devices;
        std::vector<BlockStreamLayerState> layers;
        std::vector<ggml_backend_buffer_t> host_staging_buffers;
        std::vector<void*> host_staging_data;
        ggml_backend_buffer_t common_staging_buffer = nullptr;
        void* common_staging_data = nullptr;
        size_t common_staging_bytes = 0;
        std::vector<ggml_backend_t> transfer_backends;
        std::vector<ggml_backend_event_t> transfer_events;
        sd::CUDAPeerCopyContext* peer_copy_context = nullptr;
        bool p2p_available = false;
        std::string transport;
        size_t activation_chunk_bytes = 0;
        size_t staging_slot_bytes = 0;
        size_t activation_bytes = 0;
        size_t cache_bytes_by_device[2] = {0, 0};

        ~BlockStreamGenerationState() {
            reset_transport();
        }

        BlockStreamGenerationState() = default;
        BlockStreamGenerationState(const BlockStreamGenerationState&) = delete;
        BlockStreamGenerationState& operator=(const BlockStreamGenerationState&) = delete;

        void reset_transport() {
            if (peer_copy_context != nullptr) {
                sd::destroy_cuda_peer_copy_context(peer_copy_context);
                peer_copy_context = nullptr;
            }
            for (auto event : transfer_events) {
                if (event != nullptr) {
                    ggml_backend_event_free(event);
                }
            }
            transfer_events.clear();
            for (auto backend : transfer_backends) {
                if (backend != nullptr) {
                    ggml_backend_free(backend);
                }
            }
            transfer_backends.clear();
            for (auto buffer : host_staging_buffers) {
                if (buffer != nullptr) {
                    ggml_backend_buffer_free(buffer);
                }
            }
            host_staging_buffers.clear();
            host_staging_data.clear();
            if (common_staging_buffer != nullptr) {
                ggml_backend_buffer_free(common_staging_buffer);
                common_staging_buffer = nullptr;
            }
            common_staging_data = nullptr;
            common_staging_bytes = 0;
        }
    };

    class BlockStreamGraphRunner final : public GGMLRunner {
    private:
        int device_index;
        MiniMaxH3Transformer3DModel* model;

        bool compute_no_return(const std::function<ggml_tensor*(GGMLRunnerContext*)>& build) {
            auto get_graph = [&]() {
                auto runner_ctx = get_context();
                auto result = build(&runner_ctx);
                if (result == nullptr) {
                    return static_cast<ggml_cgraph*>(nullptr);
                }
                auto graph = new_graph_custom(H3_BLOCK_STREAM_GRAPH_SIZE);
                ggml_build_forward_expand(graph, result);
                return graph;
            };
            auto result = GGMLRunner::compute<float>(get_graph,
                                                     1,
                                                     false,
                                                     false,
                                                     false,
                                                     true,
                                                     CrossForwardPrefetch::DISABLED);
            return result.has_value();
        }

    public:
        BlockStreamGraphRunner(int index,
                               ggml_backend_t backend,
                               MiniMaxH3Transformer3DModel* shared_model)
            : GGMLRunner(backend, nullptr),
              device_index(index),
              model(shared_model) {}

        std::string get_desc() override {
            return "minimax_h3_block_stream_cuda" + std::to_string(device_index);
        }

        size_t get_compute_context_tensor_capacity() const override {
            return H3_BLOCK_STREAM_TENSOR_CAPACITY;
        }

        bool run_modulations(ggml_tensor* timestep_embedding,
                             const std::vector<std::pair<int, ggml_tensor*>>& outputs) {
            auto get_graph = [&]() {
                auto runner_ctx = get_context();
                auto graph = new_graph_custom(H3_BLOCK_STREAM_GRAPH_SIZE);
                for (const auto& entry : outputs) {
                    auto block = model->transformer_block(entry.first);
                    if (block == nullptr || entry.second == nullptr) {
                        return static_cast<ggml_cgraph*>(nullptr);
                    }
                    auto projection = block->project_adaln(&runner_ctx, timestep_embedding);
                    auto copy = ggml_cpy(runner_ctx.ggml_ctx, projection, entry.second);
                    ggml_build_forward_expand(graph, copy);
                }
                return graph;
            };
            return GGMLRunner::compute<float>(get_graph,
                                              1,
                                              false,
                                              false,
                                              false,
                                              true,
                                              CrossForwardPrefetch::DISABLED)
                .has_value();
        }

        bool run_cache_fill_chunk(int block_index,
                                  int64_t token_start,
                                  int64_t token_end,
                                  const std::vector<TokenModulationSpan>& segments,
                                  ggml_tensor* input_buffer,
                                  ggml_tensor* timestep_modulation,
                                  ggml_tensor* rotary_embedding,
                                  ggml_tensor* key_cache,
                                  ggml_tensor* value_cache,
                                  ggml_tensor* cache_rows) {
            auto block = model->transformer_block(block_index);
            if (block == nullptr) {
                return false;
            }
            auto get_graph = [&]() {
                auto runner_ctx = get_context();
                auto input = ggml_ext_slice(runner_ctx.ggml_ctx,
                                            input_buffer,
                                            1,
                                            token_start,
                                            token_end,
                                            false);
                auto rows = ggml_ext_slice(runner_ctx.ggml_ctx,
                                           cache_rows,
                                           0,
                                           token_start,
                                           token_end,
                                           false);
                auto updated = block->fill_block_stream_kv_chunk(&runner_ctx,
                                                                  input,
                                                                  token_start,
                                                                  timestep_modulation,
                                                                  segments,
                                                                  rotary_embedding,
                                                                  key_cache,
                                                                  value_cache,
                                                                  rows);
                auto graph = new_graph_custom(H3_BLOCK_STREAM_GRAPH_SIZE);
                ggml_build_forward_expand(graph, updated.first);
                ggml_build_forward_expand(graph, updated.second);
                return graph;
            };
            return GGMLRunner::compute<float>(get_graph,
                                              1,
                                              false,
                                              false,
                                              false,
                                              true,
                                              CrossForwardPrefetch::DISABLED)
                .has_value();
        }

        bool run_chunk(int block_index,
                       int64_t token_start,
                       int64_t token_end,
                       int64_t valid_kv_tokens,
                       const std::vector<TokenModulationSpan>& segments,
                       ggml_tensor* input_buffer,
                       bool input_is_full_sequence,
                       ggml_tensor* output_buffer,
                       bool output_is_full_sequence,
                       ggml_tensor* timestep_modulation,
                       ggml_tensor* rotary_embedding,
                       ggml_tensor* key_cache,
                       ggml_tensor* value_cache,
                       ggml_tensor* cache_rows,
                       bool update_kv_cache = true) {
            auto block = model->transformer_block(block_index);
            if (block == nullptr) {
                return false;
            }
            return compute_no_return([&](GGMLRunnerContext* ctx) {
                const int64_t tokens = token_end - token_start;
                ggml_tensor* input = nullptr;
                if (input_is_full_sequence) {
                    input = ggml_ext_slice(ctx->ggml_ctx,
                                           input_buffer,
                                           1,
                                           token_start,
                                           token_end,
                                           false);
                } else {
                    input = ggml_view_4d(ctx->ggml_ctx,
                                         input_buffer,
                                         input_buffer->ne[0],
                                         tokens,
                                         input_buffer->ne[2],
                                         input_buffer->ne[3],
                                         input_buffer->nb[1],
                                         input_buffer->nb[2],
                                         input_buffer->nb[3],
                                         0);
                }
                auto rows = ggml_ext_slice(ctx->ggml_ctx,
                                           cache_rows,
                                           0,
                                           token_start,
                                           token_end,
                                           false);
                auto output = block->forward_block_stream_chunk(ctx,
                                                                 input,
                                                                 token_start,
                                                                 timestep_modulation,
                                                                 segments,
                                                                 rotary_embedding,
                                                                 key_cache,
                                                                 value_cache,
                                                                 rows,
                                                                 valid_kv_tokens,
                                                                 update_kv_cache);
                ggml_tensor* destination = nullptr;
                if (output_is_full_sequence) {
                    destination = ggml_view_4d(ctx->ggml_ctx,
                                               output_buffer,
                                               output_buffer->ne[0],
                                               tokens,
                                               output_buffer->ne[2],
                                               output_buffer->ne[3],
                                               output_buffer->nb[1],
                                               output_buffer->nb[2],
                                               output_buffer->nb[3],
                                               token_start * output_buffer->nb[1]);
                } else {
                    destination = ggml_view_4d(ctx->ggml_ctx,
                                               output_buffer,
                                               output_buffer->ne[0],
                                               tokens,
                                               output_buffer->ne[2],
                                               output_buffer->ne[3],
                                               output_buffer->nb[1],
                                               output_buffer->nb[2],
                                               output_buffer->nb[3],
                                               0);
                }
                return ggml_cpy(ctx->ggml_ctx, output, destination);
            });
        }

        sd::Tensor<float> run_final(ggml_tensor* hidden,
                                    ggml_tensor* timestep_embedding,
                                    const std::vector<int64_t>& video_shape,
                                    const std::vector<int64_t>& audio_shape,
                                    const TokenModulationSpan& video_segment,
                                    const TokenModulationSpan& audio_segment,
                                    float audio_slope,
                                    size_t output_dims) {
            auto get_graph = [&]() {
                auto runner_ctx = get_context();
                const int video_dims = static_cast<int>(std::min<size_t>(
                    video_shape.size(), GGML_MAX_DIMS));
                GGML_ASSERT(video_dims > 0 && audio_shape.size() >= 3);
                auto video = ggml_new_tensor(runner_ctx.ggml_ctx,
                                             GGML_TYPE_F32,
                                             video_dims,
                                             video_shape.data());
                const int64_t audio_batch = audio_shape.size() > 3
                                                ? audio_shape[3]
                                                : 1;
                auto audio = ggml_new_tensor_4d(runner_ctx.ggml_ctx,
                                                GGML_TYPE_F32,
                                                audio_shape[2],
                                                audio_shape[0] * audio_shape[1],
                                                audio_batch,
                                                1);
                auto outputs = model->forward_final(&runner_ctx,
                                                    hidden,
                                                    timestep_embedding,
                                                    video,
                                                    audio,
                                                    video_segment,
                                                    audio_segment,
                                                    audio_slope);
                const int64_t divisor = outputs.first->ne[0] * outputs.first->ne[1] * outputs.first->ne[2];
                const int64_t values  = ggml_nelements(outputs.second);
                const int64_t padding = (divisor - values % divisor) % divisor;
                auto packed_audio = ggml_reshape_4d(runner_ctx.ggml_ctx,
                                                    ggml_cont(runner_ctx.ggml_ctx, outputs.second),
                                                    values,
                                                    1,
                                                    1,
                                                    1);
                if (padding > 0) {
                    packed_audio = ggml_ext_pad(runner_ctx.ggml_ctx,
                                                packed_audio,
                                                static_cast<int>(padding),
                                                0,
                                                0,
                                                0);
                }
                packed_audio = ggml_reshape_4d(runner_ctx.ggml_ctx,
                                               packed_audio,
                                               outputs.first->ne[0],
                                               outputs.first->ne[1],
                                               outputs.first->ne[2],
                                               (values + padding) / divisor);
                auto merged = ggml_concat(runner_ctx.ggml_ctx,
                                          outputs.first,
                                          packed_audio,
                                          3);
                auto graph = new_graph_custom(H3_BLOCK_STREAM_GRAPH_SIZE);
                ggml_build_forward_expand(graph, merged);
                return graph;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(
                                                        get_graph,
                                                        1,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        CrossForwardPrefetch::DISABLED),
                                                    output_dims);
        }
    };

    class BlockStreamDeviceWorker {
    private:
        BlockStreamGraphRunner runner;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<std::function<void(BlockStreamGraphRunner&)>> tasks;
        bool stopping = false;
        std::thread thread;

        void loop() {
            for (;;) {
                std::function<void(BlockStreamGraphRunner&)> task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    condition.wait(lock, [&]() { return stopping || !tasks.empty(); });
                    if (stopping && tasks.empty()) {
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop_front();
                }
                task(runner);
            }
        }

    public:
        BlockStreamDeviceWorker(int index,
                                ggml_backend_t backend,
                                MiniMaxH3Transformer3DModel* model)
            : runner(index, backend, model),
              thread([this]() { loop(); }) {}

        ~BlockStreamDeviceWorker() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            condition.notify_all();
            if (thread.joinable()) {
                thread.join();
            }
        }

        template <typename Result>
        std::future<Result> submit(std::function<Result(BlockStreamGraphRunner&)> fn) {
            auto promise = std::make_shared<std::promise<Result>>();
            auto future = promise->get_future();
            {
                std::lock_guard<std::mutex> lock(mutex);
                tasks.push_back([promise, fn = std::move(fn)](BlockStreamGraphRunner& graph_runner) mutable {
                    try {
                        promise->set_value(fn(graph_runner));
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                });
            }
            condition.notify_one();
            return future;
        }
    };

    struct MiniMaxH3Runner : public DiffusionModelRunner {
        size_t get_compute_context_tensor_capacity() const override {
            return H3_TENSOR_CAPACITY;
        }

        struct RefinedContextCacheEntry {
            const void* context_cache_identity            = nullptr;
            const sd::Tensor<float>* source_context       = nullptr;
            const float* source_data                      = nullptr;
            std::vector<int64_t> source_shape;
            std::shared_ptr<WeightAdapter> weight_adapter = nullptr;
            ggml_context* refined_ctx                     = nullptr;
            ggml_backend_buffer_t refined_buffer          = nullptr;
            ggml_tensor* refined                          = nullptr;

            ~RefinedContextCacheEntry() {
                if (refined_buffer != nullptr) {
                    ggml_backend_buffer_free(refined_buffer);
                }
                if (refined_ctx != nullptr) {
                    ggml_free(refined_ctx);
                }
            }

            RefinedContextCacheEntry()                                           = default;
            RefinedContextCacheEntry(const RefinedContextCacheEntry&)            = delete;
            RefinedContextCacheEntry& operator=(const RefinedContextCacheEntry&) = delete;

            bool matches(const void* identity,
                         const sd::Tensor<float>& context,
                         const std::shared_ptr<WeightAdapter>& adapter) const {
                return context_cache_identity == identity &&
                       weight_adapter == adapter &&
                       source_context == &context &&
                       source_data == context.data() &&
                       source_shape == context.shape();
            }
        };

        static constexpr size_t REFINED_CONTEXT_CACHE_CAPACITY = 4;

        Config config;
        MiniMaxH3Transformer3DModel model;
        sd::Tensor<float> video_input_cache;
        sd::Tensor<float> audio_input_cache;
        sd::Tensor<float> position_input_cache;
        sd::Tensor<int32_t> modulation_index_input_cache;
        sd::Tensor<float> timestep_feature_input_cache;
        sd::Tensor<int32_t> curve_index_input_cache;
        sd::Tensor<int32_t> curve_upper_index_input_cache;
        sd::Tensor<float> curve_fraction_input_cache;
        std::vector<std::unique_ptr<RefinedContextCacheEntry>> refined_context_cache;
        std::vector<ggml_backend_t> block_stream_backends;
        std::vector<std::unique_ptr<BlockStreamDeviceWorker>> block_stream_workers;
        std::unique_ptr<BlockStreamGenerationState> block_stream_generation;
        std::vector<std::vector<ggml_tensor*>> block_stream_pinned_weights;
        bool block_stream_weights_ready = false;

        struct BlockStreamMetrics {
            double weight_io_ms = 0.0;
            double compute_ms[2] = {0.0, 0.0};
            double d2h_ms = 0.0;
            double h2d_ms = 0.0;
            double p2p_ms = 0.0;
            double scheduler_wait_ms = 0.0;
            double pipeline_bubble_ms = 0.0;
            uint64_t activation_bytes = 0;
            int64_t evaluation_count = 0;
        } block_stream_metrics;

        static Config configure(const String2TensorStorage& tensors,
                                const std::string& prefix,
                                const char* model_args,
                                bool sequence_parallel,
                                bool block_stream) {
            Config config = Config::detect_from_weights(tensors, prefix);
            config.sequence_parallel = sequence_parallel;
            config.block_stream_enabled = block_stream;
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                if (key == "minimax_h3_block_stream_warmup") {
                    if (value == "none") {
                        config.block_stream_warmup = BlockStreamWarmup::NONE;
                    } else if (value == "layer") {
                        config.block_stream_warmup = BlockStreamWarmup::LAYER;
                    } else if (value == "step") {
                        config.block_stream_warmup = BlockStreamWarmup::STEP;
                    } else {
                        config.block_stream_error = "minimax_h3_block_stream_warmup expects none, layer, or step";
                    }
                    continue;
                }
                if (key == "minimax_h3_block_stream_chunk_size") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) || parsed < 128 || parsed % 128 != 0) {
                        config.block_stream_error = "minimax_h3_block_stream_chunk_size must be a positive multiple of 128";
                    } else {
                        config.block_stream_chunk_size = parsed;
                    }
                    continue;
                }
                if (key == "minimax_h3_block_stream_stage_layers") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) || parsed < 1) {
                        config.block_stream_error = "minimax_h3_block_stream_stage_layers must be at least 1";
                    } else {
                        config.block_stream_stage_layers = parsed;
                    }
                    continue;
                }
                if (key == "minimax_h3_block_stream_buffers") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) || parsed < 2 || parsed > 16) {
                        config.block_stream_error = "minimax_h3_block_stream_buffers must be between 2 and 16";
                    } else {
                        config.block_stream_buffers = parsed;
                    }
                    continue;
                }
                if (key == "minimax_h3_block_stream_trace") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) || (parsed != 0 && parsed != 1)) {
                        config.block_stream_error = "minimax_h3_block_stream_trace expects 0 or 1";
                    } else {
                        config.block_stream_trace = parsed != 0;
                    }
                    continue;
                }
                if (key == "minimax_h3_attention_sparsity") {
                    float parsed = 0.0f;
                    if (!parse_strict_float(value, parsed) || !std::isfinite(parsed) || parsed < 0.0f || parsed >= 1.0f) {
                        LOG_WARN("ignoring invalid MiniMax-H3 model arg '%s=%s' (expected 0..less than 1)",
                                 key.c_str(),
                                 value.c_str());
                        continue;
                    }
                    config.attention_sparsity = parsed;
                    if (parsed > 0.0f) {
                        LOG_INFO("MiniMax-H3 dynamic sparse attention enabled: %.1f%% sparsity (CUDA FlashAttention only)",
                                 parsed * 100.0f);
                    }
                    continue;
                }
                if (key == "minimax_h3_mlp_chunk_size") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) || parsed < 0) {
                        LOG_WARN("ignoring invalid MiniMax-H3 model arg '%s=%s' (expected >= 0)",
                                 key.c_str(),
                                 value.c_str());
                        continue;
                    }
                    config.mlp_chunk_size = parsed;
                    continue;
                }
                if (key == "minimax_h3_attention_chunk_size") {
                    int parsed = 0;
                    if (!parse_strict_int(value, parsed) ||
                        (parsed != 0 && (parsed < 128 || parsed % 128 != 0))) {
                        LOG_WARN("ignoring invalid MiniMax-H3 model arg '%s=%s' (expected 0 or a multiple of 128)",
                                 key.c_str(),
                                 value.c_str());
                        continue;
                    }
                    config.attention_chunk_size = parsed;
                    continue;
                }
                if (key == "minimax_h3_activation_storage") {
                    if (value == "f32") {
                        config.activation_storage_type = GGML_TYPE_F32;
                    } else if (value == "bf16") {
                        config.activation_storage_type = GGML_TYPE_BF16;
                    } else if (value == "f16") {
                        config.activation_storage_type = GGML_TYPE_F16;
                    } else {
                        LOG_WARN("ignoring invalid MiniMax-H3 model arg '%s=%s' (expected f32, bf16, or f16)",
                                 key.c_str(),
                                 value.c_str());
                    }
                    continue;
                }
                if (key != "minimax_h3_layer_limit") {
                    continue;
                }
                int parsed = 0;
                if (!parse_strict_int(value, parsed) || parsed < 1 || parsed > config.num_layers) {
                    LOG_WARN("ignoring invalid MiniMax-H3 model arg '%s=%s' (expected 1..%" PRId64 ")",
                             key.c_str(),
                             value.c_str(),
                             config.num_layers);
                    continue;
                }
                config.execution_layers = parsed;
                if (config.execution_layers < config.num_layers) {
                    LOG_WARN("MiniMax-H3 benchmark layer limit is active: executing %" PRId64
                             "/%" PRId64 " transformer blocks; output quality is not representative",
                             config.execution_layers,
                             config.num_layers);
                }
            }
            if (config.mlp_chunk_size > 0) {
                LOG_INFO("MiniMax-H3 MLP token chunking enabled: chunk=%" PRId64,
                         config.mlp_chunk_size);
            }
            if (config.attention_chunk_size > 0) {
                LOG_INFO("MiniMax-H3 attention token chunking enabled: chunk=%" PRId64,
                         config.attention_chunk_size);
            }
            LOG_INFO("MiniMax-H3 activation storage: %s",
                     ggml_type_name(config.activation_storage_type));
            if (config.block_stream_enabled) {
                const char* warmup = config.block_stream_warmup == BlockStreamWarmup::NONE
                                         ? "none"
                                         : (config.block_stream_warmup == BlockStreamWarmup::LAYER ? "layer" : "step");
                LOG_INFO("MiniMax-H3 Block Stream enabled: warmup=%s chunk=%" PRId64
                         " stage_layers=%d buffers=%d trace=%s",
                         warmup,
                         config.block_stream_chunk_size,
                         config.block_stream_stage_layers,
                         config.block_stream_buffers,
                         config.block_stream_trace ? "true" : "false");
                if (!config.block_stream_error.empty()) {
                    LOG_ERROR("MiniMax-H3 Block Stream configuration error: %s",
                              config.block_stream_error.c_str());
                }
            }
            return config;
        }

        MiniMaxH3Runner(ggml_backend_t backend,
                        const String2TensorStorage& tensors,
                        const std::string& prefix                           = "model.diffusion_model",
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                         const char* model_args                              = nullptr,
                         bool sequence_parallel                              = false,
                         bool block_stream                                   = false)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(configure(tensors,
                               prefix,
                               model_args,
                               sequence_parallel,
                               block_stream)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        ~MiniMaxH3Runner() override {
            block_stream_workers.clear();
            block_stream_generation.reset();
            if (block_stream_weights_ready) {
                if (auto manager = weight_manager.lock()) {
                    for (const auto& tensors : block_stream_pinned_weights) {
                        manager->release_compute_backend_params(tensors);
                    }
                }
            }
        }

        std::string get_desc() override {
            return "minimax_h3";
        }

        bool block_stream_enabled() const {
            return config.block_stream_enabled;
        }

        int block_stream_owner(int64_t block_index) const {
            return static_cast<int>((block_index / config.block_stream_stage_layers) % 2);
        }

        int block_stream_param_owner(const std::string& name) const {
            const std::string block_prefix = prefix + ".blocks.";
            const size_t begin = name.find(block_prefix);
            if (begin != std::string::npos) {
                const size_t index_begin = begin + block_prefix.size();
                const size_t index_end = name.find('.', index_begin);
                if (index_end != std::string::npos) {
                    const int64_t index = std::strtoll(
                        name.substr(index_begin, index_end - index_begin).c_str(), nullptr, 10);
                    if (index >= 0 && index < config.execution_layers) {
                        return block_stream_owner(index);
                    }
                }
            }
            if (name.find(prefix + ".final_layer.") != std::string::npos) {
                return block_stream_owner(config.execution_layers - 1);
            }
            return 0;
        }

        bool set_block_stream_backends(const std::vector<ggml_backend_t>& backends) {
            if (!config.block_stream_enabled) {
                return backends.size() <= 1;
            }
            if (backends.size() != 2 || backends[0] == nullptr || backends[1] == nullptr ||
                backends[0] == backends[1]) {
                LOG_ERROR("MiniMax-H3 Block Stream requires exactly two distinct CUDA backends");
                return false;
            }
            for (ggml_backend_t backend : backends) {
                ggml_backend_dev_t device = ggml_backend_get_device(backend);
                ggml_backend_reg_t registry = device != nullptr
                                                  ? ggml_backend_dev_backend_reg(device)
                                                  : nullptr;
                std::string registry_name = registry != nullptr
                                                ? ggml_backend_reg_name(registry)
                                                : "";
                std::transform(registry_name.begin(),
                               registry_name.end(),
                               registry_name.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (registry_name != "cuda") {
                    LOG_ERROR("MiniMax-H3 Block Stream requires CUDA backends; got %s",
                              backend != nullptr ? ggml_backend_name(backend) : "<null>");
                    return false;
                }
            }
            block_stream_backends = backends;
            return true;
        }

        bool prepare_block_stream_weights() {
            if (!config.block_stream_enabled || block_stream_weights_ready) {
                return true;
            }
            if (block_stream_backends.size() != 2 || !config.block_stream_error.empty()) {
                LOG_ERROR("MiniMax-H3 Block Stream cannot prepare weights: %s",
                          config.block_stream_error.empty()
                              ? "two CUDA backends were not configured"
                              : config.block_stream_error.c_str());
                return false;
            }
            auto manager = weight_manager.lock();
            if (manager == nullptr) {
                LOG_ERROR("MiniMax-H3 Block Stream requires a weight manager");
                return false;
            }
            std::map<std::string, ggml_tensor*> tensors;
            model.get_param_tensors(tensors, prefix);
            block_stream_pinned_weights.assign(2, {});
            for (const auto& entry : tensors) {
                const std::string block_prefix = prefix + ".blocks.";
                const size_t block_begin = entry.first.find(block_prefix);
                const bool is_block = block_begin != std::string::npos;
                const bool is_final = entry.first.find(prefix + ".final_layer.") != std::string::npos;
                if (!is_block && !is_final) {
                    continue;
                }
                if (is_block) {
                    const size_t index_begin = block_begin + block_prefix.size();
                    const size_t index_end = entry.first.find('.', index_begin);
                    if (index_end == std::string::npos) {
                        continue;
                    }
                    const int64_t block_index = std::strtoll(
                        entry.first.substr(index_begin, index_end - index_begin).c_str(),
                        nullptr,
                        10);
                    if (block_index < 0 || block_index >= config.execution_layers) {
                        continue;
                    }
                }
                const int owner = block_stream_param_owner(entry.first);
                block_stream_pinned_weights[owner].push_back(entry.second);
            }
            const int64_t start_us = ggml_time_us();
            for (int device = 0; device < 2; ++device) {
                if (!manager->prepare_params(block_stream_pinned_weights[device])) {
                    for (int prepared = 0; prepared < device; ++prepared) {
                        manager->release_compute_backend_params(
                            block_stream_pinned_weights[prepared]);
                    }
                    LOG_ERROR("MiniMax-H3 Block Stream failed to make cuda%d block weights resident",
                              device);
                    return false;
                }
            }
            block_stream_metrics.weight_io_ms = (ggml_time_us() - start_us) / 1000.0;
            block_stream_weights_ready = true;
            LOG_INFO("MiniMax-H3 Block Stream resident weights: cuda0=%zu tensors cuda1=%zu tensors load=%.3f ms",
                     block_stream_pinned_weights[0].size(),
                     block_stream_pinned_weights[1].size(),
                     block_stream_metrics.weight_io_ms);
            return true;
        }

        bool ensure_block_stream_generation(const PackedSequenceLayout& layout,
                                            const sd::Tensor<float>& packed,
                                            const sd::Tensor<float>& video,
                                            const sd::Tensor<float>& audio,
                                            const void* conditioning_identity) {
            const int64_t sequence = static_cast<int64_t>(layout.positions.size() / 3);
            const int64_t batch = 1;
            const int64_t timestep_rows = static_cast<int64_t>(layout.timesteps.size());
            if (block_stream_generation != nullptr &&
                block_stream_generation->sequence == sequence &&
                block_stream_generation->batch == batch &&
                block_stream_generation->timestep_rows == timestep_rows &&
                block_stream_generation->conditioning_identity == conditioning_identity &&
                block_stream_generation->packed_shape == packed.shape() &&
                block_stream_generation->video_shape == video.shape() &&
                block_stream_generation->audio_shape == audio.shape()) {
                return true;
            }

            block_stream_workers.clear();
            block_stream_generation.reset();
            if (block_stream_backends.size() != 2 || !prepare_block_stream_weights()) {
                return false;
            }
            auto state = std::make_unique<BlockStreamGenerationState>();
            state->sequence = sequence;
            state->batch = batch;
            state->timestep_rows = timestep_rows;
            state->chunk_size = config.block_stream_chunk_size;
            state->chunk_count = static_cast<int>((sequence + state->chunk_size - 1) /
                                                  state->chunk_size);
            state->final_owner = block_stream_owner(config.execution_layers - 1);
            state->conditioning_identity = conditioning_identity;
            state->packed_shape = packed.shape();
            state->video_shape = video.shape();
            state->audio_shape = audio.shape();
            state->arenas.resize(2);
            state->devices.resize(2);
            state->layers.resize(static_cast<size_t>(config.execution_layers));

            const size_t tensor_capacity = 64 +
                                           static_cast<size_t>(config.execution_layers) * 4 +
                                           static_cast<size_t>(config.block_stream_buffers) * 8;
            for (int device = 0; device < 2; ++device) {
                state->arenas[device] = std::make_unique<BlockStreamPersistentArena>();
                if (!state->arenas[device]->begin(block_stream_backends[device], tensor_capacity)) {
                    LOG_ERROR("MiniMax-H3 Block Stream failed to create cuda%d persistent context",
                              device);
                    return false;
                }
                auto& arena = *state->arenas[device];
                auto& tensors = state->devices[device];
                tensors.timestep_embedding = arena.tensor_2d(GGML_TYPE_F32,
                                                              config.time_embed_dim,
                                                              timestep_rows,
                                                              device == 0
                                                                  ? "h3.block_stream.cuda0.t_emb"
                                                                  : "h3.block_stream.cuda1.t_emb");
                tensors.rotary_embedding = arena.tensor_4d(GGML_TYPE_F32,
                                                            2,
                                                            2,
                                                            config.rope_inv_freq_len * 3,
                                                            sequence,
                                                            device == 0
                                                                ? "h3.block_stream.cuda0.rope"
                                                                : "h3.block_stream.cuda1.rope");
                tensors.cache_rows = arena.tensor_1d(GGML_TYPE_I32,
                                                      sequence,
                                                      device == 0
                                                          ? "h3.block_stream.cuda0.rows"
                                                          : "h3.block_stream.cuda1.rows");
                if (device == 0) {
                    tensors.hidden_input = arena.tensor_4d(config.activation_storage_type,
                                                            config.hidden_size,
                                                            sequence,
                                                            batch,
                                                            1,
                                                            "h3.block_stream.input");
                }
                if (device == state->final_owner) {
                    tensors.final_hidden = arena.tensor_4d(config.activation_storage_type,
                                                            config.hidden_size,
                                                            sequence,
                                                            batch,
                                                            1,
                                                            "h3.block_stream.final_hidden");
                }
                if (config.block_stream_warmup != BlockStreamWarmup::NONE) {
                    if (tensors.hidden_input != nullptr) {
                        tensors.full_sequence_buffers.push_back(tensors.hidden_input);
                    }
                    if (tensors.final_hidden != nullptr &&
                        tensors.final_hidden != tensors.hidden_input) {
                        tensors.full_sequence_buffers.push_back(tensors.final_hidden);
                    }
                    while (tensors.full_sequence_buffers.size() < 2) {
                        const std::string name = "h3.block_stream.cuda" +
                                                 std::to_string(device) + ".warmup_full." +
                                                 std::to_string(tensors.full_sequence_buffers.size());
                        tensors.full_sequence_buffers.push_back(
                            arena.tensor_4d(config.activation_storage_type,
                                            config.hidden_size,
                                            sequence,
                                            batch,
                                            1,
                                            name.c_str()));
                    }
                }

                const int64_t last_tokens = sequence -
                                            static_cast<int64_t>(state->chunk_count - 1) *
                                                state->chunk_size;
                for (int slot = 0; slot < config.block_stream_buffers; ++slot) {
                    const std::string send_name = "h3.block_stream.cuda" +
                                                  std::to_string(device) + ".send." +
                                                  std::to_string(slot);
                    const std::string receive_name = "h3.block_stream.cuda" +
                                                     std::to_string(device) + ".receive." +
                                                     std::to_string(slot);
                    auto send = arena.tensor_4d(config.activation_storage_type,
                                                config.hidden_size,
                                                state->chunk_size,
                                                batch,
                                                1,
                                                send_name.c_str());
                    auto receive = arena.tensor_4d(config.activation_storage_type,
                                                   config.hidden_size,
                                                   state->chunk_size,
                                                   batch,
                                                   1,
                                                   receive_name.c_str());
                    tensors.send_buffers.push_back(send);
                    tensors.receive_buffers.push_back(receive);
                    tensors.send_last_buffers.push_back(ggml_view_4d(arena.context,
                                                                     send,
                                                                     config.hidden_size,
                                                                     last_tokens,
                                                                     batch,
                                                                     1,
                                                                     send->nb[1],
                                                                     send->nb[2],
                                                                     send->nb[3],
                                                                     0));
                    tensors.receive_last_buffers.push_back(ggml_view_4d(arena.context,
                                                                        receive,
                                                                        config.hidden_size,
                                                                        last_tokens,
                                                                        batch,
                                                                        1,
                                                                        receive->nb[1],
                                                                        receive->nb[2],
                                                                        receive->nb[3],
                                                                        0));
                }
            }

            for (int64_t block_index = 0; block_index < config.execution_layers; ++block_index) {
                const int owner = block_stream_owner(block_index);
                auto& arena = *state->arenas[owner];
                auto& layer = state->layers[static_cast<size_t>(block_index)];
                layer.owner_device = owner;
                const std::string base = "h3.block_stream.block." + std::to_string(block_index);
                layer.key_cache = arena.tensor_4d(GGML_TYPE_F16,
                                                   config.attention_head_dim,
                                                   sequence,
                                                   config.num_attention_heads,
                                                   batch,
                                                   (base + ".key").c_str());
                layer.value_cache = arena.tensor_4d(GGML_TYPE_F16,
                                                     config.attention_head_dim,
                                                     sequence,
                                                     config.num_attention_heads,
                                                     batch,
                                                     (base + ".value").c_str());
                layer.modulation = arena.tensor_2d(GGML_TYPE_F32,
                                                    config.hidden_size * 6 * 3,
                                                    timestep_rows,
                                                    (base + ".modulation").c_str());
                state->cache_bytes_by_device[owner] += ggml_nbytes(layer.key_cache) +
                                                       ggml_nbytes(layer.value_cache);
            }

            for (int device = 0; device < 2; ++device) {
                if (!state->arenas[device]->allocate()) {
                    LOG_ERROR("MiniMax-H3 Block Stream persistent allocation failed on cuda%d",
                              device);
                    return false;
                }
                std::vector<int32_t> rows(static_cast<size_t>(sequence));
                for (int64_t token = 0; token < sequence; ++token) {
                    rows[static_cast<size_t>(token)] = static_cast<int32_t>(token);
                }
                ggml_backend_tensor_set(state->devices[device].cache_rows,
                                        rows.data(),
                                        0,
                                        rows.size() * sizeof(int32_t));
            }
            state->activation_chunk_bytes = ggml_nbytes(state->devices[0].send_buffers[0]);
            state->staging_slot_bytes = std::max(
                state->activation_chunk_bytes,
                ggml_nbytes(state->devices[0].timestep_embedding));

            const auto peer = sd::enable_cuda_peer_access(block_stream_backends[0],
                                                          block_stream_backends[1]);
            state->p2p_available = peer.available && peer.enabled;
            std::string peer_copy_error;
            if (state->p2p_available) {
                state->peer_copy_context = sd::create_cuda_peer_copy_context(
                    block_stream_backends[0],
                    block_stream_backends[1],
                    &peer_copy_error);
                if (state->peer_copy_context == nullptr) {
                    state->p2p_available = false;
                }
            }
            state->transport = state->p2p_available ? "cudaMemcpyPeerAsync"
                                                    : "pinned-host-staging";
            state->transfer_backends.resize(2, nullptr);
            state->transfer_events.resize(2, nullptr);
            for (int device = 0; device < 2; ++device) {
                ggml_backend_dev_t backend_device = ggml_backend_get_device(block_stream_backends[device]);
                state->transfer_backends[device] = ggml_backend_dev_init(backend_device, nullptr);
                state->transfer_events[device] = ggml_backend_event_new(backend_device);
                if (state->transfer_backends[device] == nullptr ||
                    state->transfer_events[device] == nullptr) {
                    LOG_ERROR("MiniMax-H3 Block Stream failed to create cuda%d transfer stream/event",
                              device);
                    return false;
                }
            }
            if (!state->p2p_available) {
                state->host_staging_buffers.resize(2, nullptr);
                state->host_staging_data.resize(2, nullptr);
                const size_t staging_bytes = state->staging_slot_bytes *
                                             static_cast<size_t>(config.block_stream_buffers);
                for (int direction = 0; direction < 2; ++direction) {
                    ggml_backend_dev_t source_device = ggml_backend_get_device(
                        block_stream_backends[direction]);
                    ggml_backend_buffer_type_t host_type = ggml_backend_dev_host_buffer_type(source_device);
                    if (host_type == nullptr) {
                        LOG_ERROR("MiniMax-H3 Block Stream cuda%d has no pinned host buffer type",
                                  direction);
                        return false;
                    }
                    auto host_buffer = ggml_backend_buft_alloc_buffer(host_type, staging_bytes);
                    if (host_buffer == nullptr || ggml_backend_buffer_get_base(host_buffer) == nullptr) {
                        LOG_ERROR("MiniMax-H3 Block Stream failed to allocate pinned staging for cuda%d",
                                  direction);
                        return false;
                    }
                    state->host_staging_buffers[direction] = host_buffer;
                    state->host_staging_data[direction] = ggml_backend_buffer_get_base(host_buffer);
                }
                state->common_staging_bytes = std::max(
                    ggml_nbytes(state->devices[0].timestep_embedding),
                    ggml_nbytes(state->devices[0].rotary_embedding));
                ggml_backend_buffer_type_t host_type = ggml_backend_dev_host_buffer_type(
                    ggml_backend_get_device(block_stream_backends[0]));
                state->common_staging_buffer = ggml_backend_buft_alloc_buffer(
                    host_type,
                    state->common_staging_bytes);
                if (state->common_staging_buffer == nullptr ||
                    ggml_backend_buffer_get_base(state->common_staging_buffer) == nullptr) {
                    LOG_ERROR("MiniMax-H3 Block Stream failed to allocate pinned common-input staging");
                    return false;
                }
                state->common_staging_data = ggml_backend_buffer_get_base(
                    state->common_staging_buffer);
            }

            if (block_stream_workers.empty()) {
                for (int device = 0; device < 2; ++device) {
                    block_stream_workers.push_back(std::make_unique<BlockStreamDeviceWorker>(
                        device,
                        block_stream_backends[device],
                        &model));
                }
            }

            std::ostringstream cuda_blocks[2];
            size_t loaded_blocks[2] = {0, 0};
            for (int64_t block_index = 0; block_index < config.execution_layers; ++block_index) {
                const int owner = block_stream_owner(block_index);
                ++loaded_blocks[owner];
                if (cuda_blocks[owner].tellp() > 0) {
                    cuda_blocks[owner] << ',';
                }
                cuda_blocks[owner] << block_index;
            }
            const char* warmup = config.block_stream_warmup == BlockStreamWarmup::NONE
                                     ? "none"
                                     : (config.block_stream_warmup == BlockStreamWarmup::LAYER
                                            ? "layer"
                                            : "step");
            LOG_INFO("MiniMax-H3 Block Stream:\n"
                     "  warmup=%s\n"
                     "  sequence=%" PRId64 "\n"
                     "  chunks=%d\n"
                     "  chunk_size=%" PRId64 "\n"
                     "  blocks=%" PRId64 "\n"
                     "  stage_layers=%d\n"
                     "  cuda0_blocks=%s\n"
                     "  cuda1_blocks=%s\n"
                     "  kv_cache_cuda0=%.2f MiB\n"
                     "  kv_cache_cuda1=%.2f MiB\n"
                     "  activation_chunk=%.2f MiB\n"
                     "  transfer_buffers=%d\n"
                     "  p2p=%s\n"
                     "  transport=%s\n"
                     "  weights=disk-source/owner-resident\n"
                     "  loaded_blocks_cuda0=%zu\n"
                     "  loaded_blocks_cuda1=%zu",
                     warmup,
                     sequence,
                     state->chunk_count,
                     state->chunk_size,
                     config.execution_layers,
                     config.block_stream_stage_layers,
                     cuda_blocks[0].str().c_str(),
                     cuda_blocks[1].str().c_str(),
                     state->cache_bytes_by_device[0] / (1024.0 * 1024.0),
                     state->cache_bytes_by_device[1] / (1024.0 * 1024.0),
                     state->activation_chunk_bytes / (1024.0 * 1024.0),
                     config.block_stream_buffers,
                     state->p2p_available ? "yes" : "no",
                     state->transport.c_str(),
                     loaded_blocks[0],
                     loaded_blocks[1]);
            if (!state->p2p_available && !peer.error.empty()) {
                LOG_INFO("MiniMax-H3 Block Stream P2P unavailable: %s", peer.error.c_str());
            }
            if (!state->p2p_available && !peer_copy_error.empty()) {
                LOG_INFO("MiniMax-H3 Block Stream P2P transport unavailable: %s",
                         peer_copy_error.c_str());
            }
            block_stream_generation = std::move(state);
            return true;
        }

        bool copy_block_stream_bytes(int source_device,
                                     int destination_device,
                                     int slot,
                                     ggml_tensor* source,
                                     size_t source_offset,
                                     ggml_tensor* destination,
                                     size_t destination_offset,
                                     size_t bytes) {
            auto& state = *block_stream_generation;
            if (source_device == destination_device || source == nullptr ||
                destination == nullptr || bytes == 0 ||
                source_offset + bytes > ggml_nbytes(source) ||
                destination_offset + bytes > ggml_nbytes(destination)) {
                LOG_ERROR("MiniMax-H3 Block Stream received an invalid transfer range");
                return false;
            }
            const int64_t start_us = ggml_time_us();
            if (state.p2p_available) {
                std::string error;
                const auto direction = source_device == 0
                                           ? sd::CUDAPeerCopyDirection::A_TO_B
                                           : sd::CUDAPeerCopyDirection::B_TO_A;
                if (!sd::cuda_peer_copy(state.peer_copy_context,
                                        direction,
                                        static_cast<const char*>(source->data) + source_offset,
                                        static_cast<char*>(destination->data) + destination_offset,
                                        bytes,
                                        &error)) {
                    LOG_ERROR("MiniMax-H3 Block Stream P2P copy failed: %s", error.c_str());
                    return false;
                }
                block_stream_metrics.p2p_ms += (ggml_time_us() - start_us) / 1000.0;
            } else {
                char* host = nullptr;
                if (slot < 0) {
                    if (bytes > state.common_staging_bytes ||
                        state.common_staging_data == nullptr) {
                        LOG_ERROR("MiniMax-H3 Block Stream common-input staging is too small");
                        return false;
                    }
                    host = static_cast<char*>(state.common_staging_data);
                } else {
                    host = static_cast<char*>(state.host_staging_data[source_device]) +
                           static_cast<size_t>(slot) * state.staging_slot_bytes;
                }
                std::string device_error;
                if (!sd::select_cuda_backend_device(state.transfer_backends[source_device],
                                                    &device_error)) {
                    LOG_ERROR("MiniMax-H3 Block Stream source-device selection failed: %s",
                              device_error.c_str());
                    return false;
                }
                ggml_backend_tensor_get_async(state.transfer_backends[source_device],
                                              source,
                                              host,
                                              source_offset,
                                              bytes);
                ggml_backend_event_record(state.transfer_events[source_device],
                                          state.transfer_backends[source_device]);
                ggml_backend_event_synchronize(state.transfer_events[source_device]);
                const int64_t d2h_end_us = ggml_time_us();
                if (!sd::select_cuda_backend_device(state.transfer_backends[destination_device],
                                                    &device_error)) {
                    LOG_ERROR("MiniMax-H3 Block Stream destination-device selection failed: %s",
                              device_error.c_str());
                    return false;
                }
                ggml_backend_tensor_set_async(state.transfer_backends[destination_device],
                                              destination,
                                              host,
                                              destination_offset,
                                              bytes);
                ggml_backend_event_record(state.transfer_events[destination_device],
                                          state.transfer_backends[destination_device]);
                ggml_backend_event_synchronize(state.transfer_events[destination_device]);
                block_stream_metrics.d2h_ms += (d2h_end_us - start_us) / 1000.0;
                block_stream_metrics.h2d_ms += (ggml_time_us() - d2h_end_us) / 1000.0;
            }
            block_stream_metrics.activation_bytes += bytes;
            return true;
        }

        bool copy_block_stream_tensor(int source_device,
                                      int destination_device,
                                      int slot,
                                      ggml_tensor* source,
                                      ggml_tensor* destination) {
            if (source == nullptr || destination == nullptr ||
                source->type != destination->type ||
                !ggml_are_same_shape(source, destination) ||
                !ggml_are_same_stride(source, destination) ||
                !ggml_is_contiguous(source) ||
                !ggml_is_contiguous(destination)) {
                LOG_ERROR("MiniMax-H3 Block Stream received incompatible activation buffers");
                return false;
            }
            return copy_block_stream_bytes(source_device,
                                           destination_device,
                                           slot,
                                           source,
                                           0,
                                           destination,
                                           0,
                                           ggml_nbytes(source));
        }

        bool copy_block_stream_full_sequence(int source_device,
                                             int destination_device,
                                             ggml_tensor* source,
                                             ggml_tensor* destination) {
            auto& state = *block_stream_generation;
            if (source == nullptr || destination == nullptr ||
                source->type != destination->type ||
                !ggml_are_same_shape(source, destination) ||
                !ggml_are_same_stride(source, destination) ||
                !ggml_is_contiguous(source) ||
                !ggml_is_contiguous(destination)) {
                LOG_ERROR("MiniMax-H3 Block Stream received incompatible full-sequence buffers");
                return false;
            }
            for (int chunk = 0; chunk < state.chunk_count; ++chunk) {
                const int64_t start = static_cast<int64_t>(chunk) * state.chunk_size;
                const int64_t end = std::min(start + state.chunk_size, state.sequence);
                const size_t offset = static_cast<size_t>(start) * source->nb[1];
                const size_t bytes = static_cast<size_t>(end - start) * source->nb[1];
                if (!copy_block_stream_bytes(source_device,
                                             destination_device,
                                             chunk % config.block_stream_buffers,
                                             source,
                                             offset,
                                             destination,
                                             offset,
                                             bytes)) {
                    return false;
                }
            }
            return true;
        }

        bool run_block_stream_modulations() {
            auto& state = *block_stream_generation;
            std::future<bool> futures[2];
            for (int device = 0; device < 2; ++device) {
                std::vector<std::pair<int, ggml_tensor*>> outputs;
                for (int block = 0; block < config.execution_layers; ++block) {
                    if (state.layers[static_cast<size_t>(block)].owner_device == device) {
                        outputs.push_back({block,
                                           state.layers[static_cast<size_t>(block)].modulation});
                    }
                }
                auto timestep = state.devices[device].timestep_embedding;
                futures[device] = block_stream_workers[device]->submit<bool>(
                    [timestep, outputs = std::move(outputs)](BlockStreamGraphRunner& runner) {
                        return runner.run_modulations(timestep, outputs);
                    });
            }
            return futures[0].get() && futures[1].get();
        }

        bool run_block_stream_full_blocks(
            const std::vector<TokenModulationSpan>& segments,
            int first_block,
            int end_block,
            ggml_tensor*& hidden,
            int& hidden_owner) {
            auto& state = *block_stream_generation;
            if (first_block < 0 || end_block > config.execution_layers ||
                first_block >= end_block || hidden == nullptr) {
                return false;
            }
            const int64_t warmup_start_us = ggml_time_us();
            for (int block = first_block; block < end_block; ++block) {
                const int owner = block_stream_owner(block);
                auto& buffers = state.devices[owner].full_sequence_buffers;
                if (buffers.size() < 2) {
                    LOG_ERROR("MiniMax-H3 Block Stream warmup buffers are unavailable on cuda%d",
                              owner);
                    return false;
                }
                if (hidden_owner != owner) {
                    ggml_tensor* destination = buffers.front();
                    if (!copy_block_stream_full_sequence(hidden_owner,
                                                         owner,
                                                         hidden,
                                                         destination)) {
                        return false;
                    }
                    hidden = destination;
                    hidden_owner = owner;
                }
                ggml_tensor* output = buffers[0] == hidden ? buffers[1] : buffers[0];
                auto& layer = state.layers[static_cast<size_t>(block)];
                layer.cache_valid = false;
                layer.fresh_prefix_end = 0;
                auto* modulation = layer.modulation;
                auto* rope = state.devices[owner].rotary_embedding;
                auto* key = layer.key_cache;
                auto* value = layer.value_cache;
                auto* rows = state.devices[owner].cache_rows;
                const int64_t sequence = state.sequence;

                for (int chunk = 0; chunk < state.chunk_count; ++chunk) {
                    const int64_t start = static_cast<int64_t>(chunk) * state.chunk_size;
                    const int64_t end = std::min(start + state.chunk_size, state.sequence);
                    const int64_t started_us = ggml_time_us();
                    auto future = block_stream_workers[owner]->submit<bool>(
                        [=, &segments](BlockStreamGraphRunner& runner) {
                            return runner.run_cache_fill_chunk(block,
                                                               start,
                                                               end,
                                                               segments,
                                                               hidden,
                                                               modulation,
                                                               rope,
                                                               key,
                                                               value,
                                                               rows);
                        });
                    if (!future.get()) {
                        LOG_ERROR("MiniMax-H3 Block Stream warmup K/V fill failed at B%d/C%d",
                                  block,
                                  chunk);
                        return false;
                    }
                    block_stream_metrics.compute_ms[owner] +=
                        (ggml_time_us() - started_us) / 1000.0;
                }
                layer.cache_valid = true;
                layer.fresh_prefix_end = state.sequence;

                for (int chunk = 0; chunk < state.chunk_count; ++chunk) {
                    const int64_t start = static_cast<int64_t>(chunk) * state.chunk_size;
                    const int64_t end = std::min(start + state.chunk_size, state.sequence);
                    const int64_t started_us = ggml_time_us();
                    auto future = block_stream_workers[owner]->submit<bool>(
                        [=, &segments](BlockStreamGraphRunner& runner) {
                            return runner.run_chunk(block,
                                                    start,
                                                    end,
                                                    sequence,
                                                    segments,
                                                    hidden,
                                                    true,
                                                    output,
                                                    true,
                                                    modulation,
                                                    rope,
                                                    key,
                                                    value,
                                                    rows,
                                                    false);
                        });
                    if (!future.get()) {
                        layer.cache_valid = false;
                        LOG_ERROR("MiniMax-H3 Block Stream warmup block compute failed at B%d/C%d",
                                  block,
                                  chunk);
                        return false;
                    }
                    block_stream_metrics.compute_ms[owner] +=
                        (ggml_time_us() - started_us) / 1000.0;
                }
                hidden = output;
                hidden_owner = owner;
            }
            LOG_INFO("MiniMax-H3 Block Stream algorithmic warmup: blocks=%d..%d time=%.3f ms",
                     first_block,
                     end_block - 1,
                     (ggml_time_us() - warmup_start_us) / 1000.0);
            return true;
        }

        bool run_block_stream_wave(const std::vector<TokenModulationSpan>& segments,
                                   int first_block,
                                   ggml_tensor* initial_hidden,
                                   ggml_tensor** final_hidden_out) {
            struct InputRef {
                ggml_tensor* tensor = nullptr;
                bool full_sequence = false;
                int slot_owner = -1;
                int slot = -1;
            };
            struct TaskCompletion {
                bool ok = false;
                int block = -1;
                int chunk = -1;
                int owner = -1;
                int output_slot = -1;
                int input_slot_owner = -1;
                int input_slot = -1;
                int64_t token_start = 0;
                int64_t token_end = 0;
                int64_t started_us = 0;
                int64_t ended_us = 0;
            };
            struct ActiveTask {
                bool active = false;
                std::future<TaskCompletion> future;
            };

            auto& state = *block_stream_generation;
            const int blocks = static_cast<int>(config.execution_layers);
            const int chunks = state.chunk_count;
            if (first_block < 0 || first_block >= blocks || initial_hidden == nullptr ||
                block_stream_owner(first_block) < 0) {
                return false;
            }
            ggml_tensor* final_output = state.devices[state.final_owner].final_hidden;
            if (first_block + 1 == blocks && initial_hidden == final_output) {
                for (auto* candidate : state.devices[state.final_owner].full_sequence_buffers) {
                    if (candidate != initial_hidden) {
                        final_output = candidate;
                        break;
                    }
                }
            }
            if (final_output == nullptr || final_output == initial_hidden) {
                LOG_ERROR("MiniMax-H3 Block Stream has no distinct final output buffer");
                return false;
            }
            const int task_count = (blocks - first_block) * chunks;
            std::vector<std::vector<InputRef>> inputs(
                static_cast<size_t>(blocks),
                std::vector<InputRef>(static_cast<size_t>(chunks)));
            std::vector<std::vector<bool>> scheduled(
                static_cast<size_t>(blocks),
                std::vector<bool>(static_cast<size_t>(chunks), false));
            std::vector<std::vector<bool>> done(
                static_cast<size_t>(blocks),
                std::vector<bool>(static_cast<size_t>(chunks), false));
            std::vector<std::vector<bool>> slot_busy(
                2,
                std::vector<bool>(static_cast<size_t>(config.block_stream_buffers), false));
            for (int chunk = 0; chunk < chunks; ++chunk) {
                inputs[first_block][chunk] = {initial_hidden, true, -1, -1};
            }
            for (int block = first_block; block < blocks; ++block) {
                state.layers[static_cast<size_t>(block)].fresh_prefix_end = 0;
            }

            std::vector<TaskCompletion> trace;
            trace.reserve(static_cast<size_t>(task_count));
            ActiveTask active[2];
            int completed = 0;
            bool failed = false;
            const int64_t wave_start_us = ggml_time_us();
            const double d2h_start_ms = block_stream_metrics.d2h_ms;
            const double h2d_start_ms = block_stream_metrics.h2d_ms;
            const double p2p_start_ms = block_stream_metrics.p2p_ms;
            const double scheduler_wait_start_ms = block_stream_metrics.scheduler_wait_ms;

            auto free_slot = [&](int owner, int slot) {
                if (owner >= 0 && slot >= 0) {
                    slot_busy[owner][slot] = false;
                }
            };
            auto acquire_slot = [&](int owner) {
                for (int slot = 0; slot < config.block_stream_buffers; ++slot) {
                    if (!slot_busy[owner][slot]) {
                        slot_busy[owner][slot] = true;
                        return slot;
                    }
                }
                return -1;
            };

            auto dispatch = [&](int device) {
                if (active[device].active) {
                    return false;
                }
                for (int block = blocks - 1; block >= first_block; --block) {
                    if (block_stream_owner(block) != device) {
                        continue;
                    }
                    for (int chunk = 0; chunk < chunks; ++chunk) {
                        if (scheduled[block][chunk] || inputs[block][chunk].tensor == nullptr ||
                            (chunk > 0 && !done[block][chunk - 1])) {
                            continue;
                        }
                        int output_slot = -1;
                        if (block + 1 < blocks) {
                            output_slot = acquire_slot(device);
                            if (output_slot < 0) {
                                continue;
                            }
                        }
                        const int64_t start = static_cast<int64_t>(chunk) * state.chunk_size;
                        const int64_t end = std::min(start + state.chunk_size, state.sequence);
                        const bool cache_valid = state.layers[block].cache_valid;
                        const int64_t valid_kv_tokens = cache_valid ? state.sequence : end;
                        const InputRef input = inputs[block][chunk];
                        ggml_tensor* output = block + 1 == blocks
                                                  ? final_output
                                                  : state.devices[device].send_buffers[output_slot];
                        auto* modulation = state.layers[block].modulation;
                        auto* rope = state.devices[device].rotary_embedding;
                        auto* key = state.layers[block].key_cache;
                        auto* value = state.layers[block].value_cache;
                        auto* rows = state.devices[device].cache_rows;
                        scheduled[block][chunk] = true;
                        active[device].active = true;
                        active[device].future = block_stream_workers[device]->submit<TaskCompletion>(
                            [=, &segments](BlockStreamGraphRunner& runner) {
                                TaskCompletion result;
                                result.block = block;
                                result.chunk = chunk;
                                result.owner = device;
                                result.output_slot = output_slot;
                                result.input_slot_owner = input.slot_owner;
                                result.input_slot = input.slot;
                                result.token_start = start;
                                result.token_end = end;
                                result.started_us = ggml_time_us();
                                result.ok = runner.run_chunk(block,
                                                             start,
                                                             end,
                                                             valid_kv_tokens,
                                                             segments,
                                                             input.tensor,
                                                             input.full_sequence,
                                                             output,
                                                             block + 1 == blocks,
                                                             modulation,
                                                             rope,
                                                             key,
                                                             value,
                                                             rows);
                                result.ended_us = ggml_time_us();
                                return result;
                            });
                        return true;
                    }
                }
                return false;
            };

            while (completed < task_count && !failed) {
                bool progressed = dispatch(0) | dispatch(1);
                bool any_ready = false;
                for (int device = 0; device < 2; ++device) {
                    if (!active[device].active ||
                        active[device].future.wait_for(std::chrono::microseconds(0)) !=
                            std::future_status::ready) {
                        continue;
                    }
                    any_ready = true;
                    TaskCompletion result = active[device].future.get();
                    active[device].active = false;
                    if (!result.ok) {
                        failed = true;
                        break;
                    }
                    block_stream_metrics.compute_ms[device] +=
                        (result.ended_us - result.started_us) / 1000.0;
                    done[result.block][result.chunk] = true;
                    auto& layer = state.layers[static_cast<size_t>(result.block)];
                    layer.fresh_prefix_end = result.token_end;
                    if (result.token_end == state.sequence) {
                        layer.cache_valid = true;
                    }
                    free_slot(result.input_slot_owner, result.input_slot);
                    ++completed;
                    trace.push_back(result);

                    if (result.block + 1 < blocks) {
                        const int destination_device = block_stream_owner(result.block + 1);
                        ggml_tensor* next_input = nullptr;
                        if (destination_device == device) {
                            next_input = state.devices[device].send_buffers[result.output_slot];
                        } else {
                            dispatch(device);
                            const bool is_last_chunk = result.chunk + 1 == chunks;
                            auto* source = is_last_chunk
                                               ? state.devices[device].send_last_buffers[result.output_slot]
                                               : state.devices[device].send_buffers[result.output_slot];
                            auto* destination = is_last_chunk
                                                    ? state.devices[destination_device]
                                                          .receive_last_buffers[result.output_slot]
                                                    : state.devices[destination_device]
                                                          .receive_buffers[result.output_slot];
                            if (!copy_block_stream_tensor(device,
                                                          destination_device,
                                                          result.output_slot,
                                                          source,
                                                          destination)) {
                                failed = true;
                                break;
                            }
                            next_input = state.devices[destination_device]
                                             .receive_buffers[result.output_slot];
                        }
                        inputs[result.block + 1][result.chunk] = {
                            next_input, false, device, result.output_slot};
                        if (destination_device == device) {
                            dispatch(device);
                        }
                    }
                    progressed = true;
                }
                if (failed) {
                    break;
                }
                if (!any_ready && !progressed) {
                    const int64_t wait_start_us = ggml_time_us();
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    block_stream_metrics.scheduler_wait_ms +=
                        (ggml_time_us() - wait_start_us) / 1000.0;
                }
                if (!active[0].active && !active[1].active && !progressed &&
                    completed < task_count) {
                    LOG_ERROR("MiniMax-H3 Block Stream scheduler deadlocked after %d/%d tasks",
                              completed,
                              task_count);
                    failed = true;
                }
            }
            if (failed) {
                for (int device = 0; device < 2; ++device) {
                    if (active[device].active) {
                        active[device].future.wait();
                    }
                }
                return false;
            }

            const double wave_ms = (ggml_time_us() - wave_start_us) / 1000.0;
            const double eval_compute0 = std::accumulate(
                trace.begin(), trace.end(), 0.0,
                [](double total, const TaskCompletion& item) {
                    return total + (item.owner == 0 ? item.ended_us - item.started_us : 0) / 1000.0;
                });
            const double eval_compute1 = std::accumulate(
                trace.begin(), trace.end(), 0.0,
                [](double total, const TaskCompletion& item) {
                    return total + (item.owner == 1 ? item.ended_us - item.started_us : 0) / 1000.0;
                });
            if (config.block_stream_trace && first_block == 0 && blocks > 1) {
                bool first_overlap = false;
                int64_t b0_last_end = 0;
                int64_t b1_c0_start = INT64_MAX;
                for (const auto& item : trace) {
                    LOG_INFO("MiniMax-H3 Block Stream trace: %s B%d/C%d t=%.3fms",
                             item.started_us == item.ended_us ? "task" : "run",
                             item.block,
                             item.chunk,
                             (item.started_us - wave_start_us) / 1000.0);
                    if (item.block == 0 && item.chunk + 1 == chunks) {
                        b0_last_end = item.ended_us;
                    }
                    if (item.block == 1 && item.chunk == 0) {
                        b1_c0_start = item.started_us;
                    }
                }
                first_overlap = b1_c0_start < b0_last_end;
                LOG_INFO("MiniMax-H3 Block Stream overlap proof: start(B1,C0)=%.3fms end(B0,last)=%.3fms pass=%s",
                         (b1_c0_start - wave_start_us) / 1000.0,
                         (b0_last_end - wave_start_us) / 1000.0,
                         first_overlap ? "yes" : "no");
            }
            block_stream_metrics.pipeline_bubble_ms +=
                std::max(0.0, wave_ms * 2.0 - eval_compute0 - eval_compute1);
            const double eval_copy_ms =
                (block_stream_metrics.d2h_ms - d2h_start_ms) +
                (block_stream_metrics.h2d_ms - h2d_start_ms) +
                (block_stream_metrics.p2p_ms - p2p_start_ms);
            const double eval_scheduler_wait_ms =
                block_stream_metrics.scheduler_wait_ms - scheduler_wait_start_ms;
            LOG_INFO("MiniMax-H3 Block Stream evaluation: wave=%.3f ms cuda0=%.3f ms cuda1=%.3f ms copy=%.3f ms scheduler_wait=%.3f ms",
                     wave_ms,
                     eval_compute0,
                     eval_compute1,
                     eval_copy_ms,
                     eval_scheduler_wait_ms);
            if (final_hidden_out != nullptr) {
                *final_hidden_out = final_output;
            }
            return true;
        }

        bool reuse_compute_buffer_between_segments() const override {
            return true;
        }

        bool get_attention_sparsity(float* sparsity) const override {
            if (sparsity == nullptr) {
                return false;
            }
            *sparsity = config.attention_sparsity;
            return true;
        }

        bool set_attention_sparsity(float sparsity) override {
            if (!std::isfinite(sparsity) || sparsity < 0.0f || sparsity >= 1.0f) {
                return false;
            }
            config.attention_sparsity = sparsity;
            model.set_attention_sparsity(sparsity);
            return true;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        std::unique_ptr<RefinedContextCacheEntry> create_refined_context_cache_entry(
            const sd::Tensor<float>& context,
            const void* context_cache_identity) {
            auto entry                    = std::make_unique<RefinedContextCacheEntry>();
            entry->context_cache_identity = context_cache_identity;
            entry->source_context         = &context;
            entry->source_data            = context.data();
            entry->source_shape           = context.shape();
            entry->weight_adapter         = weight_adapter;

            auto refined_shape = context.shape();
            refined_shape[0]   = config.hidden_size;
            ggml_init_params params;
            params.mem_size    = ggml_tensor_overhead();
            params.mem_buffer  = nullptr;
            params.no_alloc    = true;
            entry->refined_ctx = ggml_init(params);
            GGML_ASSERT(entry->refined_ctx != nullptr);
            entry->refined = ggml_new_tensor(entry->refined_ctx,
                                             GGML_TYPE_F32,
                                             static_cast<int>(refined_shape.size()),
                                             refined_shape.data());
            ggml_set_name(entry->refined, "minimax_h3.refined_context");
            entry->refined_buffer = ggml_backend_alloc_ctx_tensors(entry->refined_ctx,
                                                                   runtime_backend);
            GGML_ASSERT(entry->refined_buffer != nullptr);
            ggml_backend_buffer_set_usage(entry->refined_buffer,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            return entry;
        }

        ggml_cgraph* build_context_refinement_graph(const sd::Tensor<float>& context,
                                                    ggml_tensor* refined_output) {
            GGML_ASSERT(!context.empty() && context.shape()[0] == config.text_dim);
            GGML_ASSERT(refined_output != nullptr && refined_output->ne[0] == config.hidden_size);
            auto context_input = make_input(context);
            auto runner_ctx    = get_context();
            auto refined       = model.refine_context(&runner_ctx, context_input, false);
            // Refinement graph buffers are transient; persist only their final output.
            auto output = ggml_cpy(runner_ctx.ggml_ctx, refined, refined_output);
            auto graph  = new_graph_custom(H3_GRAPH_SIZE);
            ggml_build_forward_expand(graph, output);
            return graph;
        }

        ggml_tensor* get_refined_context(const sd::Tensor<float>& context,
                                         const void* context_cache_identity,
                                         int n_threads) {
            GGML_ASSERT(!context.empty());
            GGML_ASSERT(context_cache_identity != nullptr);
            GGML_ASSERT(context.shape()[0] == config.text_dim ||
                        context.shape()[0] == config.hidden_size);

            for (const auto& entry : refined_context_cache) {
                if (entry->matches(context_cache_identity, context, weight_adapter)) {
                    return entry->refined;
                }
            }

            auto entry = create_refined_context_cache_entry(context, context_cache_identity);
            if (context.shape()[0] == config.hidden_size) {
                ggml_backend_tensor_set(entry->refined,
                                        context.data(),
                                        0,
                                        ggml_nbytes(entry->refined));
                ggml_backend_synchronize(runtime_backend);
            } else {
                auto get_graph = [&]() {
                    return build_context_refinement_graph(context, entry->refined);
                };
                auto result = GGMLRunner::compute<float>(get_graph,
                                                         n_threads,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         CrossForwardPrefetch::DISABLED);
                if (!result.has_value()) {
                    return nullptr;
                }
            }

            auto refined = entry->refined;
            if (refined_context_cache.size() == REFINED_CONTEXT_CACHE_CAPACITY) {
                refined_context_cache.erase(refined_context_cache.begin());
            }
            refined_context_cache.push_back(std::move(entry));
            return refined;
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> split_av_latents(const sd::Tensor<float>& packed,
                                                                         int audio_length) const {
            GGML_ASSERT(packed.dim() == 4 || packed.dim() == 5);
            int64_t spatial      = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
            int64_t video_values = spatial * config.video_latent_channels;
            sd::Tensor<float> video({packed.shape()[0],
                                     packed.shape()[1],
                                     packed.shape()[2],
                                     config.video_latent_channels,
                                     1});
            std::copy_n(packed.data(), static_cast<size_t>(video_values), video.data());
            if (audio_length <= 0) {
                return {video, {}};
            }
            int64_t audio_values = audio_length * 2 * config.audio_latent_channels;
            GGML_ASSERT(packed.numel() >= video_values + audio_values);
            sd::Tensor<float> audio({audio_length, 2, config.audio_latent_channels, 1});
            std::copy_n(packed.data() + video_values,
                        static_cast<size_t>(audio_values),
                        audio.data());
            return {video, audio};
        }

        ggml_tensor* merge_av_latents(ggml_context* ctx,
                                      ggml_tensor* video,
                                      ggml_tensor* audio) const {
            int64_t divisor = video->ne[0] * video->ne[1] * video->ne[2];
            int64_t values  = ggml_nelements(audio);
            int64_t padding = (divisor - values % divisor) % divisor;
            audio           = ggml_reshape_4d(ctx, ggml_cont(ctx, audio), values, 1, 1, 1);
            if (padding > 0) {
                audio = ggml_ext_pad(ctx, audio, static_cast<int>(padding), 0, 0, 0);
            }
            audio = ggml_reshape_4d(ctx,
                                    audio,
                                    video->ne[0],
                                    video->ne[1],
                                    video->ne[2],
                                    (values + padding) / divisor);
            return ggml_concat(ctx, video, audio, 3);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& packed,
                                 const sd::Tensor<float>& timestep,
                                 const sd::Tensor<float>& context_tensor,
                                 ggml_tensor* refined_context,
                                 const std::vector<sd::Tensor<float>>& condition_videos,
                                 const std::vector<sd::Tensor<float>>& condition_audios,
                                 const sd::Tensor<int32_t>& text_tags,
                                 const sd::Tensor<int32_t>& keyframe_indices,
                                 const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                 int audio_length,
                                 float video_shift,
                                 float audio_shift) {
            auto split        = split_av_latents(packed, audio_length);
            video_input_cache = std::move(split.first);
            audio_input_cache = std::move(split.second);
            GGML_ASSERT(!audio_input_cache.empty());
            GGML_ASSERT(!context_tensor.empty());
            GGML_ASSERT(refined_context != nullptr &&
                        refined_context->ne[0] == config.hidden_size);

            auto video   = make_input(video_input_cache);
            auto audio   = make_input(audio_input_cache);
            auto context = refined_context;
            std::vector<ggml_tensor*> condition_inputs;
            condition_inputs.reserve(condition_videos.size());
            for (const auto& condition : condition_videos) {
                condition_inputs.push_back(make_input(condition));
            }
            std::vector<ggml_tensor*> audio_condition_inputs;
            audio_condition_inputs.reserve(condition_audios.size());
            for (const auto& condition : condition_audios) {
                audio_condition_inputs.push_back(make_input(condition));
            }

            float sigma_v = std::clamp(timestep[0] / 1000.f, 1e-6f, 1.f);
            float t_v     = 1.f - sigma_v;
            float t_a     = 1.f - time_shift_sigma(sigma_v, video_shift, audio_shift);
            auto layout   = build_layout(context_tensor.shape()[1],
                                         video_input_cache.shape()[2],
                                         video_input_cache.shape()[1],
                                         video_input_cache.shape()[0],
                                         audio_length,
                                         condition_videos,
                                         condition_audios,
                                         keyframe_indices,
                                         reference_blocks,
                                         text_tags,
                                         t_v,
                                         t_a);

            position_input_cache = sd::Tensor<float>(
                {3, static_cast<int64_t>(layout.positions.size() / 3)},
                layout.positions);
            auto positions = make_input(position_input_cache);

            std::vector<int32_t> modulation_indices(layout.positions.size() / 3, -1);
            for (const auto& segment : layout.segments) {
                GGML_ASSERT(segment.start >= 0 && segment.end <= static_cast<int64_t>(modulation_indices.size()));
                std::fill(modulation_indices.begin() + segment.start,
                          modulation_indices.begin() + segment.end,
                          static_cast<int32_t>(segment.modulation_row));
            }
            GGML_ASSERT(std::find(modulation_indices.begin(), modulation_indices.end(), -1) ==
                        modulation_indices.end());
            modulation_index_input_cache = sd::Tensor<int32_t>(
                {static_cast<int64_t>(modulation_indices.size())},
                modulation_indices);
            auto modulation_index_input = make_input(modulation_index_input_cache);

            ggml_tensor* timestep_features   = nullptr;
            ggml_tensor* curve_indices       = nullptr;
            ggml_tensor* curve_upper_indices = nullptr;
            ggml_tensor* curve_fractions     = nullptr;
            if (config.uses_adaln_curves()) {
                std::vector<int32_t> indices(layout.timesteps.size());
                std::vector<int32_t> upper_indices(layout.timesteps.size());
                std::vector<float> fractions(layout.timesteps.size());
                for (size_t i = 0; i < layout.timesteps.size(); ++i) {
                    float position   = std::clamp(layout.timesteps[i], 0.f, 1.f) * (config.adaln_curve_grid - 1);
                    int index        = std::min(static_cast<int>(std::floor(position)),
                                                static_cast<int>(config.adaln_curve_grid - 2));
                    indices[i]       = index;
                    upper_indices[i] = index + 1;
                    fractions[i]     = position - index;
                }
                curve_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(indices.size())},
                    indices);
                curve_upper_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(upper_indices.size())},
                    upper_indices);
                curve_fraction_input_cache = sd::Tensor<float>(
                    {1, static_cast<int64_t>(fractions.size())},
                    fractions);
                curve_indices       = make_input(curve_index_input_cache);
                curve_upper_indices = make_input(curve_upper_index_input_cache);
                curve_fractions     = make_input(curve_fraction_input_cache);
            } else {
                timestep_feature_input_cache = sd::Tensor<float>(
                    {config.timestep_input_dim, static_cast<int64_t>(layout.timesteps.size())},
                    timestep_embedding(layout.timesteps,
                                       static_cast<int>(config.timestep_input_dim),
                                       10000,
                                       true,
                                       1.f));
                timestep_features = make_input(timestep_feature_input_cache);
            }

            auto runner_ctx = get_context();
            auto output     = model.forward(&runner_ctx,
                                            video,
                                            audio,
                                            context,
                                            condition_inputs,
                                            audio_condition_inputs,
                                            positions,
                                            modulation_index_input,
                                            timestep_features,
                                            curve_indices,
                                            curve_upper_indices,
                                            curve_fractions,
                                            layout.segments,
                                            layout.sequence_segments,
                                            layout.video_segment,
                                            layout.audio_segment,
                                            time_shift_slope(sigma_v, video_shift, audio_shift));
            auto merged     = merge_av_latents(compute_ctx, output.first, output.second);
            auto graph      = new_graph_custom(H3_GRAPH_SIZE);
            ggml_build_forward_expand(graph, merged);
            return graph;
        }

        sd::Tensor<float> compute_block_stream(
            int n_threads,
            const sd::Tensor<float>& packed,
            const sd::Tensor<float>& timestep,
            const sd::Tensor<float>& context_tensor,
            ggml_tensor* refined_context,
            const std::vector<sd::Tensor<float>>& condition_videos,
            const std::vector<sd::Tensor<float>>& condition_audios,
            const sd::Tensor<int32_t>& text_tags,
            const sd::Tensor<int32_t>& keyframe_indices,
            const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
            int audio_length,
            float video_shift,
            float audio_shift,
            const void* conditioning_identity) {
            if (!config.block_stream_error.empty() || block_stream_backends.size() != 2) {
                LOG_ERROR("MiniMax-H3 Block Stream unavailable: %s",
                          config.block_stream_error.empty()
                              ? "exactly two CUDA backends are required"
                              : config.block_stream_error.c_str());
                return {};
            }
            if (!flash_attn_enabled) {
                LOG_ERROR("MiniMax-H3 Block Stream requires diffusion FlashAttention");
                return {};
            }
            if (config.attention_sparsity != 0.0f) {
                LOG_ERROR("MiniMax-H3 Block Stream does not support dynamic sparse attention");
                return {};
            }
            if (weight_adapter != nullptr) {
                LOG_ERROR("MiniMax-H3 Block Stream does not support LoRA adapters");
                return {};
            }
            if (packed.shape().empty() || packed.shape().back() != 1) {
                LOG_ERROR("MiniMax-H3 Block Stream currently requires batch=1");
                return {};
            }
            auto split = split_av_latents(packed, audio_length);
            video_input_cache = std::move(split.first);
            audio_input_cache = std::move(split.second);
            if (audio_input_cache.empty() || context_tensor.empty() || refined_context == nullptr) {
                LOG_ERROR("MiniMax-H3 Block Stream received incomplete audio/context inputs");
                return {};
            }

            const float sigma_v = std::clamp(timestep[0] / 1000.f, 1e-6f, 1.f);
            const float t_v = 1.f - sigma_v;
            const float t_a = 1.f - time_shift_sigma(sigma_v, video_shift, audio_shift);
            const auto layout = build_layout(context_tensor.shape()[1],
                                             video_input_cache.shape()[2],
                                             video_input_cache.shape()[1],
                                             video_input_cache.shape()[0],
                                             audio_length,
                                             condition_videos,
                                             condition_audios,
                                             keyframe_indices,
                                             reference_blocks,
                                             text_tags,
                                             t_v,
                                             t_a,
                                             true);
            if (!ensure_block_stream_generation(layout,
                                                packed,
                                                video_input_cache,
                                                audio_input_cache,
                                                conditioning_identity)) {
                return {};
            }
            auto& state = *block_stream_generation;

            position_input_cache = sd::Tensor<float>(
                {3, static_cast<int64_t>(layout.positions.size() / 3)},
                layout.positions);
            std::vector<int32_t> modulation_indices(layout.positions.size() / 3, -1);
            for (const auto& segment : layout.segments) {
                std::fill(modulation_indices.begin() + segment.start,
                          modulation_indices.begin() + segment.end,
                          static_cast<int32_t>(segment.modulation_row));
            }
            if (std::find(modulation_indices.begin(), modulation_indices.end(), -1) !=
                modulation_indices.end()) {
                LOG_ERROR("MiniMax-H3 Block Stream modulation layout is incomplete");
                return {};
            }
            modulation_index_input_cache = sd::Tensor<int32_t>(
                {static_cast<int64_t>(modulation_indices.size())},
                modulation_indices);

            if (config.uses_adaln_curves()) {
                std::vector<int32_t> indices(layout.timesteps.size());
                std::vector<int32_t> upper_indices(layout.timesteps.size());
                std::vector<float> fractions(layout.timesteps.size());
                for (size_t i = 0; i < layout.timesteps.size(); ++i) {
                    const float position = std::clamp(layout.timesteps[i], 0.f, 1.f) *
                                           (config.adaln_curve_grid - 1);
                    const int index = std::min(static_cast<int>(std::floor(position)),
                                               static_cast<int>(config.adaln_curve_grid - 2));
                    indices[i] = index;
                    upper_indices[i] = index + 1;
                    fractions[i] = position - index;
                }
                curve_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(indices.size())}, indices);
                curve_upper_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(upper_indices.size())}, upper_indices);
                curve_fraction_input_cache = sd::Tensor<float>(
                    {1, static_cast<int64_t>(fractions.size())}, fractions);
            } else {
                timestep_feature_input_cache = sd::Tensor<float>(
                    {config.timestep_input_dim, static_cast<int64_t>(layout.timesteps.size())},
                    timestep_embedding(layout.timesteps,
                                       static_cast<int>(config.timestep_input_dim),
                                       10000,
                                       true,
                                       1.f));
            }

            auto prefix_graph = [&]() {
                auto video = make_input(video_input_cache);
                auto audio = make_input(audio_input_cache);
                std::vector<ggml_tensor*> condition_inputs;
                condition_inputs.reserve(condition_videos.size());
                for (const auto& condition : condition_videos) {
                    condition_inputs.push_back(make_input(condition));
                }
                std::vector<ggml_tensor*> audio_condition_inputs;
                audio_condition_inputs.reserve(condition_audios.size());
                for (const auto& condition : condition_audios) {
                    audio_condition_inputs.push_back(make_input(condition));
                }
                auto positions = make_input(position_input_cache);
                ggml_tensor* timestep_features = nullptr;
                ggml_tensor* curve_indices = nullptr;
                ggml_tensor* curve_upper_indices = nullptr;
                ggml_tensor* curve_fractions = nullptr;
                if (config.uses_adaln_curves()) {
                    curve_indices = make_input(curve_index_input_cache);
                    curve_upper_indices = make_input(curve_upper_index_input_cache);
                    curve_fractions = make_input(curve_fraction_input_cache);
                } else {
                    timestep_features = make_input(timestep_feature_input_cache);
                }

                auto runner_ctx = get_context();
                auto prepared = model.prepare_block_inputs(&runner_ctx,
                                                           video,
                                                           audio,
                                                           refined_context,
                                                           condition_inputs,
                                                           audio_condition_inputs,
                                                           positions,
                                                           timestep_features,
                                                           curve_indices,
                                                           curve_upper_indices,
                                                           curve_fractions,
                                                           layout.sequence_segments);
                auto hidden_copy = ggml_cpy(runner_ctx.ggml_ctx,
                                            prepared.hidden,
                                            state.devices[0].hidden_input);
                auto timestep_copy = ggml_cpy(runner_ctx.ggml_ctx,
                                              prepared.timestep_embedding,
                                              state.devices[0].timestep_embedding);
                auto rope_copy = ggml_cpy(runner_ctx.ggml_ctx,
                                          prepared.rotary_embedding,
                                          state.devices[0].rotary_embedding);
                auto graph = new_graph_custom(H3_GRAPH_SIZE);
                ggml_build_forward_expand(graph, hidden_copy);
                ggml_build_forward_expand(graph, timestep_copy);
                ggml_build_forward_expand(graph, rope_copy);
                return graph;
            };
            auto prefix_result = GGMLRunner::compute<float>(prefix_graph,
                                                            n_threads,
                                                            false,
                                                            true,
                                                            false,
                                                            true,
                                                            CrossForwardPrefetch::DISABLED);
            if (!prefix_result.has_value()) {
                LOG_ERROR("MiniMax-H3 Block Stream input-pack graph failed");
                return {};
            }
            if (!copy_block_stream_tensor(0,
                                          1,
                                          -1,
                                          state.devices[0].timestep_embedding,
                                          state.devices[1].timestep_embedding) ||
                !copy_block_stream_tensor(0,
                                          1,
                                          -1,
                                          state.devices[0].rotary_embedding,
                                          state.devices[1].rotary_embedding)) {
                return {};
            }
            if (!run_block_stream_modulations()) {
                return {};
            }

            const int final_owner = state.final_owner;
            ggml_tensor* final_hidden = state.devices[final_owner].final_hidden;
            const bool first_evaluation = state.evaluation_count == 0;
            if (first_evaluation &&
                config.block_stream_warmup == BlockStreamWarmup::STEP) {
                int hidden_owner = 0;
                final_hidden = state.devices[0].hidden_input;
                if (!run_block_stream_full_blocks(layout.segments,
                                                  0,
                                                  static_cast<int>(config.execution_layers),
                                                  final_hidden,
                                                  hidden_owner) ||
                    hidden_owner != final_owner) {
                    return {};
                }
            } else if (first_evaluation &&
                       config.block_stream_warmup == BlockStreamWarmup::LAYER) {
                int hidden_owner = 0;
                ggml_tensor* hidden = state.devices[0].hidden_input;
                if (!run_block_stream_full_blocks(layout.segments,
                                                  0,
                                                  1,
                                                  hidden,
                                                  hidden_owner)) {
                    return {};
                }
                if (config.execution_layers == 1) {
                    final_hidden = hidden;
                } else {
                    const int first_stream_owner = block_stream_owner(1);
                    if (hidden_owner != first_stream_owner) {
                        auto& buffers = state.devices[first_stream_owner].full_sequence_buffers;
                        ggml_tensor* destination = buffers.front();
                        if (config.execution_layers == 2 &&
                            destination == state.devices[first_stream_owner].final_hidden) {
                            destination = buffers[1];
                        }
                        if (!copy_block_stream_full_sequence(hidden_owner,
                                                             first_stream_owner,
                                                             hidden,
                                                             destination)) {
                            return {};
                        }
                        hidden = destination;
                    }
                    if (!run_block_stream_wave(layout.segments,
                                               1,
                                               hidden,
                                               &final_hidden)) {
                        return {};
                    }
                }
            } else if (!run_block_stream_wave(layout.segments,
                                              0,
                                              state.devices[0].hidden_input,
                                              &final_hidden)) {
                return {};
            }

            auto final_future = block_stream_workers[final_owner]->submit<sd::Tensor<float>>(
                [&](BlockStreamGraphRunner& runner) {
                    return runner.run_final(final_hidden,
                                            state.devices[final_owner].timestep_embedding,
                                            state.video_shape,
                                            state.audio_shape,
                                            layout.video_segment,
                                            layout.audio_segment,
                                            time_shift_slope(sigma_v, video_shift, audio_shift),
                                            packed.dim());
                });
            auto result = final_future.get();
            if (result.empty()) {
                LOG_ERROR("MiniMax-H3 Block Stream final projection failed");
                return {};
            }
            for (int64_t i = 0; i < result.numel(); ++i) {
                if (!std::isfinite(result.data()[i])) {
                    LOG_ERROR("MiniMax-H3 Block Stream produced NaN/Inf at output index %" PRId64,
                              i);
                    return {};
                }
            }
            ++state.evaluation_count;
            ++block_stream_metrics.evaluation_count;
            return result;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& params) override {
            GGML_ASSERT(params.x != nullptr && params.timesteps != nullptr && params.context != nullptr);
            const auto* extra = diffusion_extra_as<MiniMaxH3DiffusionExtra>(params);
            static const std::vector<sd::Tensor<float>> empty_conditions;
            static const std::vector<MiniMaxH3ReferenceBlock> empty_reference_blocks;
            const auto& conditions       = params.ref_latents == nullptr ? empty_conditions : *params.ref_latents;
            const auto& audio_conditions = extra->reference_audio_latents == nullptr
                                               ? empty_conditions
                                               : *extra->reference_audio_latents;
            const auto& reference_blocks = extra->reference_blocks == nullptr
                                               ? empty_reference_blocks
                                               : *extra->reference_blocks;
            const sd::Tensor<int32_t> empty_int;
            const void* context_cache_identity = params.context_cache_identity != nullptr
                                                     ? params.context_cache_identity
                                                     : params.context;
            auto context = get_refined_context(*params.context,
                                               context_cache_identity,
                                               n_threads);
            if (context == nullptr) {
                return {};
            }
            if (config.block_stream_enabled) {
                return compute_block_stream(
                    n_threads,
                    *params.x,
                    *params.timesteps,
                    *params.context,
                    context,
                    conditions,
                    audio_conditions,
                    extra->text_token_tags == nullptr ? empty_int : *extra->text_token_tags,
                    extra->keyframe_indices == nullptr ? empty_int : *extra->keyframe_indices,
                    reference_blocks,
                    extra->audio_length,
                    extra->video_sigma_shift,
                    extra->audio_sigma_shift,
                    context_cache_identity);
            }
            auto get_graph = [&]() {
                return build_graph(*params.x,
                                   *params.timesteps,
                                   *params.context,
                                   context,
                                   conditions,
                                   audio_conditions,
                                   extra->text_token_tags == nullptr ? empty_int : *extra->text_token_tags,
                                   extra->keyframe_indices == nullptr ? empty_int : *extra->keyframe_indices,
                                   reference_blocks,
                                   extra->audio_length,
                                   extra->video_sigma_shift,
                                   extra->audio_sigma_shift);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph,
                                                                              n_threads,
                                                                              false,
                                                                              false,
                                                                              false),
                                                   params.x->dim());
        }

    protected:
        void on_sampling_done() override {
            if (config.block_stream_enabled) {
                LOG_INFO("MiniMax-H3 Block Stream sampling metrics: evaluations=%" PRId64
                         " weight_io=%.3fms cuda0_compute=%.3fms cuda1_compute=%.3fms"
                         " activation_d2h=%.3fms activation_h2d=%.3fms activation_p2p=%.3fms"
                         " activation_bytes=%" PRIu64 " scheduler_wait=%.3fms pipeline_bubble=%.3fms",
                         block_stream_metrics.evaluation_count,
                         block_stream_metrics.weight_io_ms,
                         block_stream_metrics.compute_ms[0],
                         block_stream_metrics.compute_ms[1],
                         block_stream_metrics.d2h_ms,
                         block_stream_metrics.h2d_ms,
                         block_stream_metrics.p2p_ms,
                         block_stream_metrics.activation_bytes,
                         block_stream_metrics.scheduler_wait_ms,
                         block_stream_metrics.pipeline_bubble_ms);
                block_stream_workers.clear();
                block_stream_generation.reset();
                const double resident_weight_io_ms = block_stream_metrics.weight_io_ms;
                block_stream_metrics = {};
                block_stream_metrics.weight_io_ms = resident_weight_io_ms;
            }
            refined_context_cache.clear();
        }
    };

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__
