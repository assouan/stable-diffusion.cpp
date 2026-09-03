#ifndef __SD_CORE_LAYER_STREAM_PREFETCH_H__
#define __SD_CORE_LAYER_STREAM_PREFETCH_H__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct RunnerWeightManager;

namespace sd::ggml_graph_cut {
    struct Plan;
}

namespace sd {
    class LayerStreamPrefetch {
    private:
        std::weak_ptr<RunnerWeightManager> manager_;
        uintptr_t owner_id_ = 0;
        std::vector<std::vector<ggml_tensor*>> segment_params_;
        std::vector<ggml_tensor*> queued_params_;
        size_t queued_segment_ = SIZE_MAX;
        bool enabled_          = true;

        size_t next_parameter_segment(size_t segment_index) const;
        void disable();

    public:
        LayerStreamPrefetch(
            const std::shared_ptr<RunnerWeightManager>& manager,
            uintptr_t owner_id,
            ggml_cgraph* graph,
            const ggml_graph_cut::Plan& plan,
            const std::unordered_set<const ggml_tensor*>& params,
            bool enabled = true);
        ~LayerStreamPrefetch();

        bool enabled() const { return enabled_; }
        bool activate(size_t segment_index);
        bool enqueue_next(size_t segment_index);
        void clear();
    };
}

#endif  // __SD_CORE_LAYER_STREAM_PREFETCH_H__
