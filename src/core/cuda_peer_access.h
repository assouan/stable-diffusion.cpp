#ifndef __SD_CORE_CUDA_PEER_ACCESS_H__
#define __SD_CORE_CUDA_PEER_ACCESS_H__

#include <cstddef>
#include <string>

#include "ggml-backend.h"

namespace sd {

    struct CUDAPeerAccessResult {
        bool available = false;
        bool enabled   = false;
        int device_a   = -1;
        int device_b   = -1;
        std::string error;
    };

    CUDAPeerAccessResult enable_cuda_peer_access(ggml_backend_t backend_a,
                                                  ggml_backend_t backend_b);

    bool select_cuda_backend_device(ggml_backend_t backend, std::string* error = nullptr);

    enum class CUDAPeerCopyDirection {
        A_TO_B,
        B_TO_A,
    };

    struct CUDAPeerCopyContext;

    CUDAPeerCopyContext* create_cuda_peer_copy_context(ggml_backend_t backend_a,
                                                        ggml_backend_t backend_b,
                                                        std::string* error = nullptr);
    void destroy_cuda_peer_copy_context(CUDAPeerCopyContext* context);
    bool cuda_peer_copy(CUDAPeerCopyContext* context,
                        CUDAPeerCopyDirection direction,
                        const void* source,
                        void* destination,
                        size_t bytes,
                        std::string* error = nullptr);

}  // namespace sd

#endif  // __SD_CORE_CUDA_PEER_ACCESS_H__
