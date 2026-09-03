#include "core/layer_stream_prefetch.h"

#include <utility>

#include "core/ggml_graph_cut.h"
#include "weight_manager.h"

namespace sd {
    static ggml_tensor* canonical_param(
        ggml_tensor* tensor,
        const std::unordered_set<const ggml_tensor*>& params) {
        for (ggml_tensor* current = tensor; current != nullptr; current = current->view_src) {
            if (params.find(current) != params.end()) {
                return current;
            }
        }
        return nullptr;
    }

    LayerStreamPrefetch::LayerStreamPrefetch(
        const std::shared_ptr<RunnerWeightManager>& manager,
        uintptr_t owner_id,
        ggml_cgraph* graph,
        const ggml_graph_cut::Plan& plan,
        const std::unordered_set<const ggml_tensor*>& params,
        bool enabled)
        : manager_(manager),
          owner_id_(owner_id),
          enabled_(enabled && manager != nullptr) {
        segment_params_.resize(plan.segments.size());
        for (size_t segment_index = 0; segment_index < plan.segments.size(); ++segment_index) {
            std::unordered_set<ggml_tensor*> seen;
            for (ggml_tensor* tensor :
                 ggml_graph_cut::param_tensors(graph, plan.segments[segment_index])) {
                ggml_tensor* param = canonical_param(tensor, params);
                if (param != nullptr && seen.insert(param).second) {
                    segment_params_[segment_index].push_back(param);
                }
            }
        }
    }

    LayerStreamPrefetch::~LayerStreamPrefetch() {
        clear();
    }

    size_t LayerStreamPrefetch::next_parameter_segment(size_t segment_index) const {
        for (size_t next = segment_index + 1; next < segment_params_.size(); ++next) {
            if (!segment_params_[next].empty()) {
                return next;
            }
        }
        return SIZE_MAX;
    }

    void LayerStreamPrefetch::disable() {
        clear();
        enabled_ = false;
    }

    bool LayerStreamPrefetch::activate(size_t segment_index) {
        if (!enabled_ || queued_segment_ == SIZE_MAX) {
            return true;
        }
        if (queued_segment_ != segment_index) {
            return true;
        }

        auto manager = manager_.lock();
        if (manager == nullptr ||
            !manager->activate_prefetched_params(owner_id_, queued_params_)) {
            disable();
            return false;
        }
        queued_params_.clear();
        queued_segment_ = SIZE_MAX;
        return true;
    }

    bool LayerStreamPrefetch::enqueue_next(size_t segment_index) {
        if (!enabled_) {
            return true;
        }
        if (queued_segment_ != SIZE_MAX) {
            return true;
        }

        const size_t next_segment = next_parameter_segment(segment_index);
        if (next_segment == SIZE_MAX) {
            return true;
        }

        std::unordered_set<ggml_tensor*> active_params(
            segment_params_[segment_index].begin(),
            segment_params_[segment_index].end());
        std::vector<ggml_tensor*> params;
        params.reserve(segment_params_[next_segment].size());
        for (ggml_tensor* param : segment_params_[next_segment]) {
            if (active_params.find(param) == active_params.end()) {
                params.push_back(param);
            }
        }
        if (params.empty()) {
            return true;
        }

        auto manager = manager_.lock();
        if (manager == nullptr || !manager->prefetch_params(owner_id_, params)) {
            disable();
            return false;
        }
        queued_params_  = std::move(params);
        queued_segment_ = next_segment;
        return true;
    }

    void LayerStreamPrefetch::clear() {
        if (auto manager = manager_.lock()) {
            manager->clear_prefetched_params(owner_id_);
        }
        queued_params_.clear();
        queued_segment_ = SIZE_MAX;
    }
}
