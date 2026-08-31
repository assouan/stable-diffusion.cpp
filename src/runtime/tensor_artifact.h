#ifndef __SD_RUNTIME_TENSOR_ARTIFACT_H__
#define __SD_RUNTIME_TENSOR_ARTIFACT_H__

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"

namespace sd {

    struct TensorArtifactView {
        std::string name;
        ggml_type type = GGML_TYPE_COUNT;
        std::vector<int64_t> shape;
        const void* data = nullptr;
    };

    struct TensorArtifactData {
        ggml_type type = GGML_TYPE_COUNT;
        std::vector<int64_t> shape;
        std::vector<uint8_t> bytes;
    };

    bool write_safetensors_artifact(const std::string& path,
                                    const std::vector<TensorArtifactView>& tensors,
                                    std::string* error = nullptr);

    bool read_safetensors_artifact(const std::string& path,
                                   std::map<std::string, TensorArtifactData>* tensors,
                                   std::string* error = nullptr);

}  // namespace sd

#endif  // __SD_RUNTIME_TENSOR_ARTIFACT_H__
