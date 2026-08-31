#include "core/cuda_peer_access.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace sd {

    using CUDAError = int;

    static constexpr CUDAError CUDA_SUCCESS                           = 0;
    static constexpr CUDAError CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704;
    static constexpr unsigned int CUDA_STREAM_NON_BLOCKING             = 1;
    static constexpr unsigned int CUDA_EVENT_DISABLE_TIMING            = 2;

    struct CUDARuntimeAPI {
        using GetDeviceCountFn          = CUDAError (*)(int*);
        using GetDeviceFn               = CUDAError (*)(int*);
        using SetDeviceFn               = CUDAError (*)(int);
        using DeviceCanAccessPeerFn      = CUDAError (*)(int*, int, int);
        using DeviceEnablePeerAccessFn   = CUDAError (*)(int, unsigned int);
        using DeviceGetPCIBusIdFn        = CUDAError (*)(char*, int, int);
        using GetErrorStringFn           = const char* (*)(CUDAError);
        using StreamCreateWithFlagsFn     = CUDAError (*)(void**, unsigned int);
        using StreamDestroyFn             = CUDAError (*)(void*);
        using EventCreateWithFlagsFn      = CUDAError (*)(void**, unsigned int);
        using EventDestroyFn              = CUDAError (*)(void*);
        using EventRecordFn               = CUDAError (*)(void*, void*);
        using EventSynchronizeFn          = CUDAError (*)(void*);
        using MemcpyPeerAsyncFn            = CUDAError (*)(void*, int, const void*, int, size_t, void*);

        void* library                                     = nullptr;
        GetDeviceCountFn get_device_count                 = nullptr;
        GetDeviceFn get_device                            = nullptr;
        SetDeviceFn set_device                            = nullptr;
        DeviceCanAccessPeerFn device_can_access_peer      = nullptr;
        DeviceEnablePeerAccessFn device_enable_peer_access = nullptr;
        DeviceGetPCIBusIdFn device_get_pci_bus_id         = nullptr;
        GetErrorStringFn get_error_string                 = nullptr;
        StreamCreateWithFlagsFn stream_create_with_flags  = nullptr;
        StreamDestroyFn stream_destroy                    = nullptr;
        EventCreateWithFlagsFn event_create_with_flags    = nullptr;
        EventDestroyFn event_destroy                      = nullptr;
        EventRecordFn event_record                        = nullptr;
        EventSynchronizeFn event_synchronize              = nullptr;
        MemcpyPeerAsyncFn memcpy_peer_async               = nullptr;
        std::string load_error;

        bool loaded() const {
            return library != nullptr && get_device_count != nullptr && get_device != nullptr &&
                   set_device != nullptr && device_can_access_peer != nullptr &&
                   device_enable_peer_access != nullptr;
        }
    };

#if defined(_WIN32)
    static void* cuda_runtime_open_library(const char* name) {
        return reinterpret_cast<void*>(LoadLibraryA(name));
    }

    static void* cuda_runtime_symbol(void* library, const char* name) {
        return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
    }
#else
    static void* cuda_runtime_open_library(const char* name) {
        return dlopen(name, RTLD_NOW | RTLD_LOCAL);
    }

    static void* cuda_runtime_symbol(void* library, const char* name) {
        return dlsym(library, name);
    }
#endif

    template <typename Function>
    static Function cuda_runtime_load_symbol(void* library, const char* name) {
        return reinterpret_cast<Function>(cuda_runtime_symbol(library, name));
    }

    static CUDARuntimeAPI load_cuda_runtime_api() {
        CUDARuntimeAPI api;
#if defined(_WIN32)
        const char* candidates[] = {
            "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64.dll"};
#else
        const char* candidates[] = {
            "libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"};
#endif
        for (const char* candidate : candidates) {
            api.library = cuda_runtime_open_library(candidate);
            if (api.library != nullptr) {
                break;
            }
        }
        if (api.library == nullptr) {
            api.load_error = "CUDA runtime library was not found";
            return api;
        }

        api.get_device_count = cuda_runtime_load_symbol<CUDARuntimeAPI::GetDeviceCountFn>(
            api.library, "cudaGetDeviceCount");
        api.get_device = cuda_runtime_load_symbol<CUDARuntimeAPI::GetDeviceFn>(api.library,
                                                                               "cudaGetDevice");
        api.set_device = cuda_runtime_load_symbol<CUDARuntimeAPI::SetDeviceFn>(api.library,
                                                                               "cudaSetDevice");
        api.device_can_access_peer = cuda_runtime_load_symbol<CUDARuntimeAPI::DeviceCanAccessPeerFn>(
            api.library, "cudaDeviceCanAccessPeer");
        api.device_enable_peer_access = cuda_runtime_load_symbol<CUDARuntimeAPI::DeviceEnablePeerAccessFn>(
            api.library, "cudaDeviceEnablePeerAccess");
        api.device_get_pci_bus_id = cuda_runtime_load_symbol<CUDARuntimeAPI::DeviceGetPCIBusIdFn>(
            api.library, "cudaDeviceGetPCIBusId");
        api.get_error_string = cuda_runtime_load_symbol<CUDARuntimeAPI::GetErrorStringFn>(
            api.library, "cudaGetErrorString");
        api.stream_create_with_flags = cuda_runtime_load_symbol<CUDARuntimeAPI::StreamCreateWithFlagsFn>(
            api.library, "cudaStreamCreateWithFlags");
        api.stream_destroy = cuda_runtime_load_symbol<CUDARuntimeAPI::StreamDestroyFn>(
            api.library, "cudaStreamDestroy");
        api.event_create_with_flags = cuda_runtime_load_symbol<CUDARuntimeAPI::EventCreateWithFlagsFn>(
            api.library, "cudaEventCreateWithFlags");
        api.event_destroy = cuda_runtime_load_symbol<CUDARuntimeAPI::EventDestroyFn>(
            api.library, "cudaEventDestroy");
        api.event_record = cuda_runtime_load_symbol<CUDARuntimeAPI::EventRecordFn>(
            api.library, "cudaEventRecord");
        api.event_synchronize = cuda_runtime_load_symbol<CUDARuntimeAPI::EventSynchronizeFn>(
            api.library, "cudaEventSynchronize");
        api.memcpy_peer_async = cuda_runtime_load_symbol<CUDARuntimeAPI::MemcpyPeerAsyncFn>(
            api.library, "cudaMemcpyPeerAsync");
        if (!api.loaded()) {
            api.load_error = "CUDA runtime does not export the peer-access API";
        }
        return api;
    }

    static CUDARuntimeAPI& cuda_runtime_api() {
        static CUDARuntimeAPI api = load_cuda_runtime_api();
        return api;
    }

    static std::string cuda_error_message(const CUDARuntimeAPI& api,
                                          const char* operation,
                                          CUDAError error) {
        std::ostringstream message;
        message << operation << " failed with CUDA error " << error;
        if (api.get_error_string != nullptr) {
            const char* text = api.get_error_string(error);
            if (text != nullptr && text[0] != '\0') {
                message << " (" << text << ")";
            }
        }
        return message.str();
    }

    static std::string lower_copy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }

    static bool is_cuda_device(ggml_backend_dev_t device) {
        if (device == nullptr) {
            return false;
        }
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const char* registry_name   = registry != nullptr ? ggml_backend_reg_name(registry) : nullptr;
        return registry_name != nullptr && lower_copy(registry_name) == "cuda";
    }

    static std::string normalized_pci_bus_id(const char* id) {
        if (id == nullptr) {
            return {};
        }
        std::string result = lower_copy(id);
        while (result.size() > 1 && result[0] == '0' && result[1] != ':') {
            result.erase(result.begin());
        }
        return result;
    }

    static int cuda_device_index_from_backend(const CUDARuntimeAPI& api,
                                              ggml_backend_t backend,
                                              std::string* error) {
        ggml_backend_dev_t device = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
        if (!is_cuda_device(device)) {
            *error = "backend is not a CUDA device";
            return -1;
        }

        int runtime_device_count = 0;
        CUDAError status         = api.get_device_count(&runtime_device_count);
        if (status != CUDA_SUCCESS) {
            *error = cuda_error_message(api, "cudaGetDeviceCount", status);
            return -1;
        }

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const std::string requested_bus_id = normalized_pci_bus_id(props.device_id);
        if (!requested_bus_id.empty() && api.device_get_pci_bus_id != nullptr) {
            for (int runtime_device = 0; runtime_device < runtime_device_count; ++runtime_device) {
                char bus_id[64] = {};
                status          = api.device_get_pci_bus_id(bus_id, sizeof(bus_id), runtime_device);
                if (status == CUDA_SUCCESS && normalized_pci_bus_id(bus_id) == requested_bus_id) {
                    return runtime_device;
                }
            }
        }

        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const size_t registry_count = registry != nullptr ? ggml_backend_reg_dev_count(registry) : 0;
        for (size_t i = 0; i < registry_count; ++i) {
            if (ggml_backend_reg_dev_get(registry, i) == device &&
                i < static_cast<size_t>(runtime_device_count)) {
                return static_cast<int>(i);
            }
        }

        *error = "could not map the ggml CUDA backend to a CUDA runtime device";
        return -1;
    }

    static bool enable_cuda_peer_direction(const CUDARuntimeAPI& api,
                                           int source_device,
                                           int peer_device,
                                           std::string* error) {
        CUDAError status = api.set_device(source_device);
        if (status != CUDA_SUCCESS) {
            *error = cuda_error_message(api, "cudaSetDevice", status);
            return false;
        }
        status = api.device_enable_peer_access(peer_device, 0);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            *error = cuda_error_message(api, "cudaDeviceEnablePeerAccess", status);
            return false;
        }
        return true;
    }

    CUDAPeerAccessResult enable_cuda_peer_access(ggml_backend_t backend_a,
                                                  ggml_backend_t backend_b) {
        CUDAPeerAccessResult result;
        CUDARuntimeAPI& api = cuda_runtime_api();
        if (!api.loaded()) {
            result.error = api.load_error;
            return result;
        }

        result.device_a = cuda_device_index_from_backend(api, backend_a, &result.error);
        if (result.device_a < 0) {
            return result;
        }
        result.device_b = cuda_device_index_from_backend(api, backend_b, &result.error);
        if (result.device_b < 0) {
            return result;
        }
        if (result.device_a == result.device_b) {
            result.error = "peer access requires two distinct CUDA devices";
            return result;
        }

        int previous_device = -1;
        CUDAError status    = api.get_device(&previous_device);
        if (status != CUDA_SUCCESS) {
            result.error = cuda_error_message(api, "cudaGetDevice", status);
            return result;
        }

        int a_to_b = 0;
        int b_to_a = 0;
        status     = api.device_can_access_peer(&a_to_b, result.device_a, result.device_b);
        if (status == CUDA_SUCCESS) {
            status = api.device_can_access_peer(&b_to_a, result.device_b, result.device_a);
        }
        if (status != CUDA_SUCCESS) {
            result.error = cuda_error_message(api, "cudaDeviceCanAccessPeer", status);
        } else if (a_to_b == 0 || b_to_a == 0) {
            result.error = "bidirectional CUDA peer access is unavailable";
        } else {
            result.available = true;
            if (enable_cuda_peer_direction(api, result.device_a, result.device_b, &result.error) &&
                enable_cuda_peer_direction(api, result.device_b, result.device_a, &result.error)) {
                result.enabled = true;
            }
        }

        CUDAError restore_status = api.set_device(previous_device);
        if (restore_status != CUDA_SUCCESS && result.error.empty()) {
            result.error = cuda_error_message(api, "cudaSetDevice(restore)", restore_status);
            result.enabled = false;
        }
        return result;
    }

    bool select_cuda_backend_device(ggml_backend_t backend, std::string* error) {
        if (error != nullptr) {
            error->clear();
        }
        CUDARuntimeAPI& api = cuda_runtime_api();
        if (!api.loaded()) {
            if (error != nullptr) {
                *error = api.load_error;
            }
            return false;
        }
        std::string mapping_error;
        const int device = cuda_device_index_from_backend(api, backend, &mapping_error);
        if (device < 0) {
            if (error != nullptr) {
                *error = std::move(mapping_error);
            }
            return false;
        }
        const CUDAError status = api.set_device(device);
        if (status != CUDA_SUCCESS) {
            if (error != nullptr) {
                *error = cuda_error_message(api, "cudaSetDevice", status);
            }
            return false;
        }
        return true;
    }

    struct CUDAPeerCopyContext {
        int devices[2] = {-1, -1};
        void* streams[2] = {nullptr, nullptr};
        void* events[2] = {nullptr, nullptr};
    };

    static bool cuda_peer_copy_api_available(const CUDARuntimeAPI& api, std::string* error) {
        if (!api.loaded()) {
            if (error != nullptr) {
                *error = api.load_error;
            }
            return false;
        }
        if (api.stream_create_with_flags == nullptr || api.stream_destroy == nullptr ||
            api.event_create_with_flags == nullptr || api.event_destroy == nullptr ||
            api.event_record == nullptr || api.event_synchronize == nullptr ||
            api.memcpy_peer_async == nullptr) {
            if (error != nullptr) {
                *error = "CUDA runtime does not export the asynchronous peer-copy API";
            }
            return false;
        }
        return true;
    }

    static void destroy_cuda_peer_copy_resources(const CUDARuntimeAPI& api,
                                                 CUDAPeerCopyContext* context) {
        if (context == nullptr) {
            return;
        }
        for (int direction = 0; direction < 2; ++direction) {
            const int source_index = direction;
            if (context->devices[source_index] >= 0) {
                (void)api.set_device(context->devices[source_index]);
            }
            if (context->events[direction] != nullptr) {
                (void)api.event_destroy(context->events[direction]);
                context->events[direction] = nullptr;
            }
            if (context->streams[direction] != nullptr) {
                (void)api.stream_destroy(context->streams[direction]);
                context->streams[direction] = nullptr;
            }
        }
    }

    CUDAPeerCopyContext* create_cuda_peer_copy_context(ggml_backend_t backend_a,
                                                        ggml_backend_t backend_b,
                                                        std::string* error) {
        if (error != nullptr) {
            error->clear();
        }
        CUDARuntimeAPI& api = cuda_runtime_api();
        if (!cuda_peer_copy_api_available(api, error)) {
            return nullptr;
        }

        const CUDAPeerAccessResult access = enable_cuda_peer_access(backend_a, backend_b);
        if (!access.available || !access.enabled) {
            if (error != nullptr) {
                *error = access.error.empty() ? "CUDA peer access is unavailable" : access.error;
            }
            return nullptr;
        }

        int previous_device = -1;
        CUDAError status = api.get_device(&previous_device);
        if (status != CUDA_SUCCESS) {
            if (error != nullptr) {
                *error = cuda_error_message(api, "cudaGetDevice", status);
            }
            return nullptr;
        }

        auto* context = new (std::nothrow) CUDAPeerCopyContext();
        if (context == nullptr) {
            if (error != nullptr) {
                *error = "failed to allocate CUDA peer-copy context";
            }
            return nullptr;
        }
        context->devices[0] = access.device_a;
        context->devices[1] = access.device_b;

        bool initialized = true;
        for (int direction = 0; direction < 2 && initialized; ++direction) {
            status = api.set_device(context->devices[direction]);
            if (status == CUDA_SUCCESS) {
                status = api.stream_create_with_flags(&context->streams[direction],
                                                       CUDA_STREAM_NON_BLOCKING);
            }
            if (status == CUDA_SUCCESS) {
                status = api.event_create_with_flags(&context->events[direction],
                                                      CUDA_EVENT_DISABLE_TIMING);
            }
            if (status != CUDA_SUCCESS) {
                initialized = false;
                if (error != nullptr) {
                    *error = cuda_error_message(api, "CUDA peer-copy resource creation", status);
                }
            }
        }

        if (!initialized) {
            destroy_cuda_peer_copy_resources(api, context);
            delete context;
            context = nullptr;
        }
        const CUDAError restore_status = api.set_device(previous_device);
        if (restore_status != CUDA_SUCCESS && context != nullptr) {
            if (error != nullptr) {
                *error = cuda_error_message(api, "cudaSetDevice(restore)", restore_status);
            }
            destroy_cuda_peer_copy_resources(api, context);
            delete context;
            context = nullptr;
        }
        return context;
    }

    void destroy_cuda_peer_copy_context(CUDAPeerCopyContext* context) {
        if (context == nullptr) {
            return;
        }
        CUDARuntimeAPI& api = cuda_runtime_api();
        int previous_device = -1;
        const bool restore = api.loaded() && api.get_device(&previous_device) == CUDA_SUCCESS;
        if (cuda_peer_copy_api_available(api, nullptr)) {
            destroy_cuda_peer_copy_resources(api, context);
        }
        if (restore) {
            (void)api.set_device(previous_device);
        }
        delete context;
    }

    bool cuda_peer_copy(CUDAPeerCopyContext* context,
                        CUDAPeerCopyDirection direction,
                        const void* source,
                        void* destination,
                        size_t bytes,
                        std::string* error) {
        if (error != nullptr) {
            error->clear();
        }
        if (context == nullptr || source == nullptr || destination == nullptr) {
            if (error != nullptr) {
                *error = "CUDA peer copy received a null context or pointer";
            }
            return false;
        }
        if (bytes == 0) {
            return true;
        }

        CUDARuntimeAPI& api = cuda_runtime_api();
        if (!cuda_peer_copy_api_available(api, error)) {
            return false;
        }
        const int direction_index = direction == CUDAPeerCopyDirection::A_TO_B ? 0 : 1;
        const int source_device = context->devices[direction_index];
        const int destination_device = context->devices[1 - direction_index];

        int previous_device = -1;
        CUDAError status = api.get_device(&previous_device);
        if (status == CUDA_SUCCESS) {
            status = api.set_device(source_device);
        }
        if (status == CUDA_SUCCESS) {
            status = api.memcpy_peer_async(destination,
                                           destination_device,
                                           source,
                                           source_device,
                                           bytes,
                                           context->streams[direction_index]);
        }
        if (status == CUDA_SUCCESS) {
            status = api.event_record(context->events[direction_index],
                                      context->streams[direction_index]);
        }
        if (status == CUDA_SUCCESS) {
            status = api.event_synchronize(context->events[direction_index]);
        }
        if (status != CUDA_SUCCESS && error != nullptr) {
            *error = cuda_error_message(api, "cudaMemcpyPeerAsync", status);
        }

        if (previous_device >= 0) {
            const CUDAError restore_status = api.set_device(previous_device);
            if (restore_status != CUDA_SUCCESS && status == CUDA_SUCCESS) {
                status = restore_status;
                if (error != nullptr) {
                    *error = cuda_error_message(api, "cudaSetDevice(restore)", restore_status);
                }
            }
        }
        return status == CUDA_SUCCESS;
    }

}  // namespace sd
