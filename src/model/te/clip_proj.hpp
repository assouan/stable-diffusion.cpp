#ifndef __SD_MODEL_TE_CLIP_PROJ_HPP__
#define __SD_MODEL_TE_CLIP_PROJ_HPP__

#include <cstdint>
#include <string>

#include "core/ggml_extend.hpp"

namespace LLM {

    struct ClipProjConfig {
        int64_t input_dim  = 0;
        int64_t output_dim = 0;
        int tap_layer      = -1;

        static bool detect(const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix,
                           ClipProjConfig& config,
                           std::string& error) {
            const std::string full_prefix = prefix.empty() ? "" : prefix + ".";
            auto find_tensor              = [&](const char* name) -> const TensorStorage* {
                auto tensor = tensor_storage_map.find(full_prefix + name);
                return tensor == tensor_storage_map.end() ? nullptr : &tensor->second;
            };

            const TensorStorage* weight = find_tensor("W");
            if (weight == nullptr || weight->n_dims != 2 || weight->ne[0] <= 0 || weight->ne[1] <= 0) {
                error = "ClipProj requires a two-dimensional W tensor";
                return false;
            }

            config.input_dim  = weight->ne[1];
            config.output_dim = weight->ne[0];

            auto validate_vector = [&](const char* name, int64_t size) {
                const TensorStorage* tensor = find_tensor(name);
                if (tensor == nullptr) {
                    error = std::string("ClipProj is missing tensor '") + name + "'";
                    return false;
                }
                if (tensor->ne[0] != size || tensor->nelements() != size) {
                    error = std::string("ClipProj tensor '") + name + "' has an invalid shape";
                    return false;
                }
                return true;
            };

            return validate_vector("mean_in", config.input_dim) &&
                   validate_vector("std_in", config.input_dim) &&
                   validate_vector("mean_out", config.output_dim) &&
                   validate_vector("std_out", config.output_dim) &&
                   validate_vector("sink_out", config.output_dim);
        }

        bool validate(int64_t encoder_dim,
                      int64_t encoder_layers,
                      int64_t target_dim,
                      std::string& error) const {
            if (input_dim != encoder_dim) {
                error = "input dimension " + std::to_string(input_dim) +
                        " does not match the LLM hidden size " + std::to_string(encoder_dim);
                return false;
            }
            if (output_dim != target_dim) {
                error = "output dimension " + std::to_string(output_dim) +
                        " does not match the MiniMax-H3 text dimension " + std::to_string(target_dim);
                return false;
            }
            if (tap_layer <= 0 || tap_layer > encoder_layers) {
                error = "tap layer " + std::to_string(tap_layer) +
                        " is outside the LLM layer range 1.." + std::to_string(encoder_layers);
                return false;
            }
            return true;
        }
    };

    class ClipProj : public UnaryBlock {
    private:
        ClipProjConfig config_;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            (void)tensor_storage_map;
            (void)prefix;
            params["W"]        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config_.output_dim, config_.input_dim);
            params["mean_in"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.input_dim);
            params["std_in"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.input_dim);
            params["mean_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.output_dim);
            params["std_out"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.output_dim);
            params["sink_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.output_dim);
        }

    public:
        explicit ClipProj(const ClipProjConfig& config)
            : config_(config) {
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(x->ne[0] == config_.input_dim);

            x = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, params["mean_in"]), params["std_in"]);
            // OUT_PROD accepts the stored x @ W orientation without materializing a transposed weight.
            auto out = ggml_out_prod(ctx->ggml_ctx, params["W"], ggml_transpose(ctx->ggml_ctx, x));
            if (!ggml_backend_supports_op(ctx->backend, out)) {
                auto weight = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, params["W"]));
                out         = ggml_mul_mat(ctx->ggml_ctx, weight, x);
            }
            out = ggml_add(ctx->ggml_ctx,
                           ggml_mul(ctx->ggml_ctx, out, params["std_out"]),
                           params["mean_out"]);

            // MiniMax-H3 uses a calibrated, prompt-independent attention-sink token.
            auto sink_shape = ggml_new_tensor_4d(ctx->ggml_ctx,
                                                 GGML_TYPE_F32,
                                                 config_.output_dim,
                                                 1,
                                                 out->ne[2],
                                                 out->ne[3]);
            auto sink       = ggml_repeat(ctx->ggml_ctx, params["sink_out"], sink_shape);
            return ggml_set(ctx->ggml_ctx, out, sink, out->nb[1], out->nb[2], out->nb[3], 0);
        }
    };

}  // namespace LLM

#endif  // __SD_MODEL_TE_CLIP_PROJ_HPP__
