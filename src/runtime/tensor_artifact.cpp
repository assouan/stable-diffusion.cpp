#include "tensor_artifact.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <thread>

#include "model_io/safetensors_io.h"
#include "model_io/tensor_storage.h"

namespace fs = std::filesystem;

namespace sd {

    static void set_artifact_error(std::string* error, const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
    }

    static bool artifact_tensor_nbytes(const TensorArtifactView& tensor,
                                       size_t* nbytes,
                                       std::string* error) {
        if (tensor.name.empty()) {
            set_artifact_error(error, "artifact tensor name is empty");
            return false;
        }
        if (tensor.type != GGML_TYPE_F32 && tensor.type != GGML_TYPE_I32) {
            set_artifact_error(error,
                               "unsupported artifact tensor type for '" + tensor.name + "'");
            return false;
        }
        if (tensor.shape.empty() || tensor.shape.size() > SD_MAX_DIMS ||
            (tensor.shape.size() == SD_MAX_DIMS && tensor.shape.back() != 1)) {
            set_artifact_error(error,
                               "artifact tensor '" + tensor.name +
                                   "' must have between 1 and 4 dimensions, or a trailing singleton fifth dimension");
            return false;
        }

        uint64_t elements = 1;
        for (int64_t dim : tensor.shape) {
            if (dim <= 0 || elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
                set_artifact_error(error,
                                   "invalid artifact tensor shape for '" + tensor.name + "'");
                return false;
            }
            elements *= static_cast<uint64_t>(dim);
        }
        const uint64_t type_size = ggml_type_size(tensor.type);
        if (elements > std::numeric_limits<size_t>::max() / type_size) {
            set_artifact_error(error,
                               "artifact tensor is too large: '" + tensor.name + "'");
            return false;
        }
        *nbytes = static_cast<size_t>(elements * type_size);
        if (*nbytes > 0 && tensor.data == nullptr) {
            set_artifact_error(error,
                               "artifact tensor data is null: '" + tensor.name + "'");
            return false;
        }
        return true;
    }

    bool write_safetensors_artifact(const std::string& path,
                                    const std::vector<TensorArtifactView>& tensors,
                                    std::string* error) {
        if (path.empty()) {
            set_artifact_error(error, "artifact output path is empty");
            return false;
        }
        if (tensors.empty()) {
            set_artifact_error(error, "artifact has no tensors");
            return false;
        }

        fs::path final_path(path);
        std::error_code ec;
        if (fs::exists(final_path, ec)) {
            set_artifact_error(error,
                               ec ? "unable to inspect artifact output path: " + ec.message()
                                  : "artifact output already exists: " + final_path.string());
            return false;
        }
        if (ec) {
            set_artifact_error(error, "unable to inspect artifact output path: " + ec.message());
            return false;
        }
        const fs::path parent = final_path.parent_path();
        if (!parent.empty() && !fs::is_directory(parent, ec)) {
            set_artifact_error(error,
                               ec ? "unable to inspect artifact output directory: " + ec.message()
                                  : "artifact output directory does not exist: " + parent.string());
            return false;
        }

        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        std::ostringstream temporary_name;
        temporary_name << final_path.filename().string() << ".tmp-" << std::hex << nonce << '-'
                       << std::hash<std::thread::id>{}(std::this_thread::get_id());
        fs::path temporary_path = parent / temporary_name.str();

        ggml_init_params context_params = {};
        context_params.mem_size = tensors.size() * ggml_tensor_overhead() + 1024;
        context_params.no_alloc = true;
        ggml_context* context   = ggml_init(context_params);
        if (context == nullptr) {
            set_artifact_error(error, "unable to allocate artifact tensor metadata");
            return false;
        }

        std::vector<TensorWriteInfo> write_infos;
        write_infos.reserve(tensors.size());
        bool valid = true;
        for (const TensorArtifactView& tensor : tensors) {
            size_t expected_nbytes = 0;
            if (!artifact_tensor_nbytes(tensor, &expected_nbytes, error)) {
                valid = false;
                break;
            }

            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (size_t i = 0; i < std::min(tensor.shape.size(), static_cast<size_t>(GGML_MAX_DIMS)); ++i) {
                ne[i] = tensor.shape[i];
            }
            if (tensor.shape.size() == SD_MAX_DIMS) {
                ne[3] *= tensor.shape[4];
            }
            ggml_tensor* ggml_tensor_view = ggml_new_tensor(context,
                                                            tensor.type,
                                                            std::min(static_cast<int>(tensor.shape.size()), GGML_MAX_DIMS),
                                                            ne);
            if (ggml_tensor_view == nullptr || ggml_nbytes(ggml_tensor_view) != expected_nbytes) {
                set_artifact_error(error,
                                   "unable to describe artifact tensor '" + tensor.name + "'");
                valid = false;
                break;
            }
            ggml_tensor_view->data = const_cast<void*>(tensor.data);
            ggml_set_name(ggml_tensor_view, tensor.name.c_str());

            TensorWriteInfo write_info;
            write_info.n_dims = static_cast<int>(tensor.shape.size());
            for (int i = 0; i < write_info.n_dims; ++i) {
                write_info.ne[i] = tensor.shape[static_cast<size_t>(i)];
            }
            write_info.tensor = ggml_tensor_view;
            write_infos.push_back(write_info);
        }

        bool written = false;
        if (valid) {
            written = write_safetensors_file(temporary_path.string(), write_infos, error);
        }
        ggml_free(context);
        if (!written) {
            fs::remove(temporary_path, ec);
            return false;
        }

        fs::rename(temporary_path, final_path, ec);
        if (ec) {
            fs::remove(temporary_path, ec);
            set_artifact_error(error, "unable to publish artifact: " + ec.message());
            return false;
        }
        return true;
    }

    bool read_safetensors_artifact(const std::string& path,
                                   std::map<std::string, TensorArtifactData>* tensors,
                                   std::string* error) {
        if (tensors == nullptr) {
            set_artifact_error(error, "artifact tensor destination is null");
            return false;
        }
        tensors->clear();

        std::vector<TensorStorage> storages;
        if (!read_safetensors_file(path, storages, error)) {
            return false;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            set_artifact_error(error, "unable to open artifact: " + path);
            return false;
        }

        for (const TensorStorage& storage : storages) {
            if (storage.type != GGML_TYPE_F32 && storage.type != GGML_TYPE_I32) {
                set_artifact_error(error,
                                   "unsupported tensor type in artifact: '" + storage.name + "'");
                tensors->clear();
                return false;
            }
            if (storage.n_dims <= 0 || storage.n_dims > GGML_MAX_DIMS) {
                set_artifact_error(error,
                                   "unsupported tensor rank in artifact: '" + storage.name + "'");
                tensors->clear();
                return false;
            }
            if (tensors->find(storage.name) != tensors->end()) {
                set_artifact_error(error,
                                   "duplicate tensor in artifact: '" + storage.name + "'");
                tensors->clear();
                return false;
            }

            TensorArtifactData tensor;
            tensor.type = storage.type;
            tensor.shape.assign(storage.ne, storage.ne + storage.n_dims);
            const int64_t signed_nbytes = storage.nbytes();
            if (signed_nbytes < 0 || static_cast<uint64_t>(signed_nbytes) > std::numeric_limits<size_t>::max()) {
                set_artifact_error(error,
                                   "invalid tensor size in artifact: '" + storage.name + "'");
                tensors->clear();
                return false;
            }
            tensor.bytes.resize(static_cast<size_t>(signed_nbytes));
            file.clear();
            file.seekg(static_cast<std::streamoff>(storage.offset), std::ios::beg);
            file.read(reinterpret_cast<char*>(tensor.bytes.data()),
                      static_cast<std::streamsize>(tensor.bytes.size()));
            if (!file) {
                set_artifact_error(error,
                                   "unable to read tensor from artifact: '" + storage.name + "'");
                tensors->clear();
                return false;
            }
            tensors->emplace(storage.name, std::move(tensor));
        }

        if (tensors->empty()) {
            set_artifact_error(error, "artifact contains no supported tensors");
            return false;
        }
        return true;
    }

}  // namespace sd
