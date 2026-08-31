// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

namespace fs = std::filesystem;

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        default:
            return "img_gen";
    }
}

const char* async_job_status_name(AsyncJobStatus status) {
    switch (status) {
        case AsyncJobStatus::Queued:
            return "queued";
        case AsyncJobStatus::Generating:
            return "generating";
        case AsyncJobStatus::Completed:
            return "completed";
        case AsyncJobStatus::Failed:
            return "failed";
        case AsyncJobStatus::Cancelled:
            return "cancelled";
        default:
            return "failed";
    }
}

void purge_expired_jobs(AsyncJobManager& manager) {
    const int64_t now = unix_timestamp_now();

    for (auto it = manager.expired_jobs.begin(); it != manager.expired_jobs.end();) {
        if (it->second <= now) {
            it = manager.expired_jobs.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = manager.jobs.begin(); it != manager.jobs.end();) {
        const auto& job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }

        int64_t ttl_seconds = job->status == AsyncJobStatus::Completed
                                  ? manager.completed_ttl_seconds
                                  : manager.failed_ttl_seconds;
        if (now - job->completed_at >= ttl_seconds) {
            manager.expired_jobs[job->id] = now + std::max<int64_t>(ttl_seconds, 60);
            it                            = manager.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

size_t count_pending_jobs(const AsyncJobManager& manager) {
    size_t pending = 0;
    for (const auto& entry : manager.jobs) {
        if (entry.second->status == AsyncJobStatus::Queued ||
            entry.second->status == AsyncJobStatus::Generating) {
            ++pending;
        }
    }
    return pending;
}

std::string make_async_job_id(AsyncJobManager& manager) {
    std::ostringstream oss;
    oss << "job_" << std::hex << unix_timestamp_now() << "_" << std::setw(8)
        << std::setfill('0') << manager.next_id++;
    return oss.str();
}

bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job) {
    auto new_end = std::remove(manager.queue.begin(), manager.queue.end(), job.id);
    if (new_end == manager.queue.end()) {
        return false;
    }

    manager.queue.erase(new_end, manager.queue.end());
    job.status       = AsyncJobStatus::Cancelled;
    job.completed_at = unix_timestamp_now();
    job.result_images_b64.clear();
    job.result_media_b64.clear();
    job.result_media_path.clear();
    job.result_media_file_name.clear();
    job.result_media_mime_type.clear();
    job.result_frame_count = 0;
    job.result_fps         = 0;
    job.error_code         = "cancelled";
    job.error_message      = "job cancelled by client";
    return true;
}

json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job) {
    json result;
    result["id"]             = job.id;
    result["kind"]           = async_job_kind_name(job.kind);
    result["status"]         = async_job_status_name(job.status);
    result["created"]        = job.created_at;
    result["started"]        = job.started_at == 0 ? json(nullptr) : json(job.started_at);
    result["completed"]      = job.completed_at == 0 ? json(nullptr) : json(job.completed_at);
    result["queue_position"] = 0;

    if (job.status == AsyncJobStatus::Queued) {
        size_t position = 1;
        for (const auto& queued_id : manager.queue) {
            if (queued_id == job.id) {
                result["queue_position"] = position;
                break;
            }
            ++position;
        }
    }

    if (job.status == AsyncJobStatus::Completed) {
        if (job.kind == AsyncJobKind::VidGen) {
            if (job.vid_gen.conditioning_only || job.vid_gen.latent_only) {
                result["result"] = json::object();
            } else {
                result["result"] = {
                    {"output_format", job.vid_gen.output_format},
                    {"mime_type", job.result_media_mime_type},
                    {"fps", job.result_fps},
                    {"frame_count", job.result_frame_count},
                };
                if (job.result_media_path.empty()) {
                    result["result"]["b64_json"] = job.result_media_b64;
                } else {
                    result["result"]["file_name"]    = job.result_media_file_name;
                    result["result"]["download_url"] = "/sdcpp/v1/jobs/" + job.id + "/result";
                }
            }
            json artifacts = json::object();
            if (!job.vid_gen.conditioning_output_file_name.empty()) {
                artifacts["conditioning"] = job.vid_gen.conditioning_output_file_name;
            }
            if (!job.vid_gen.latent_output_file_name.empty()) {
                artifacts["latent"] = job.vid_gen.latent_output_file_name;
            }
            if (!artifacts.empty()) {
                result["result"]["artifacts"] = std::move(artifacts);
            }
        } else {
            json images = json::array();
            for (size_t i = 0; i < job.result_images_b64.size(); ++i) {
                images.push_back({{"index", i}, {"b64_json", job.result_images_b64[i]}});
            }
            result["result"] = {
                {"output_format", job.img_gen.output_format},
                {"images", images},
            };
        }
        result["error"] = nullptr;
    } else if (job.status == AsyncJobStatus::Failed ||
               job.status == AsyncJobStatus::Cancelled) {
        result["result"] = nullptr;
        result["error"]  = {
             {"code",
             job.error_code.empty()
                  ? (job.status == AsyncJobStatus::Cancelled ? "cancelled" : "generation_failed")
                  : job.error_code},
             {"message", job.error_message},
        };
    } else {
        result["result"] = nullptr;
        result["error"]  = nullptr;
    }

    return result;
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::string& error_message) {
    sd_img_gen_params_t params = job.img_gen.to_sd_img_gen_params_t();

    SDImageVec results;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        int num_results         = 0;
        if (!generate_image(runtime.sd_ctx, &params, &raw_results, &num_results)) {
            raw_results = nullptr;
            num_results = 0;
        }
        results.adopt(raw_results, num_results);
    }

    const int num_results = results.count();
    if (num_results <= 0) {
        error_message = "generate_image returned no results";
        return false;
    }

    EncodedImageFormat encoded_format = EncodedImageFormat::PNG;
    if (job.img_gen.output_format == "jpeg") {
        encoded_format = EncodedImageFormat::JPEG;
    } else if (job.img_gen.output_format == "webp") {
        encoded_format = EncodedImageFormat::WEBP;
    }

    int batch_count      = job.img_gen.gen_params.batch_count;
    int images_per_batch = batch_count > 0 ? std::max(1, num_results / batch_count) : 1;
    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }

        const std::string metadata = job.img_gen.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime.ctx_params,
                                                            job.img_gen.gen_params,
                                                            job.img_gen.gen_params.seed + i / images_per_batch)
                                         : "";
        auto image_bytes           = encode_image_to_vector(encoded_format,
                                                            results[i].data,
                                                            results[i].width,
                                                            results[i].height,
                                                            results[i].channel,
                                                            metadata,
                                                            job.img_gen.output_compression);
        if (image_bytes.empty()) {
            continue;
        }
        output_images.push_back(base64_encode(image_bytes));
    }

    if (output_images.empty()) {
        error_message = "generate_image returned empty encoded outputs";
        return false;
    }

    return true;
}

static bool persist_video_output(const std::string& output_dir,
                                 const std::string& job_id,
                                 const std::string& output_format,
                                 const std::vector<uint8_t>& bytes,
                                 std::string& output_path,
                                 std::string& output_file_name,
                                 std::string& error_message) {
    std::error_code ec;
    fs::path final_path;
    for (int suffix = 0; suffix < 1000; ++suffix) {
        output_file_name = job_id;
        if (suffix > 0) {
            output_file_name += "-" + std::to_string(suffix);
        }
        output_file_name += "." + output_format;
        final_path = fs::path(output_dir) / output_file_name;
        const bool exists = fs::exists(final_path, ec);
        if (ec) {
            error_message = "unable to inspect persisted video output path";
            return false;
        }
        if (!exists) {
            break;
        }
        if (suffix == 999) {
            error_message = "unable to select a unique persisted video path";
            return false;
        }
    }

    fs::path temporary_path = final_path;
    temporary_path += ".tmp";
    std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error_message = "unable to open persisted video output";
        return false;
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) {
        fs::remove(temporary_path, ec);
        error_message = "unable to write persisted video output";
        return false;
    }

    fs::rename(temporary_path, final_path, ec);
    if (ec) {
        fs::remove(temporary_path, ec);
        error_message = "unable to publish persisted video output";
        return false;
    }
    output_path = final_path.u8string();
    return true;
}

bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_path,
                         std::string& output_media_file_name,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message) {
    sd_vid_gen_params_t params = job.vid_gen.to_sd_vid_gen_params_t();
    sd_vid_gen_artifact_params_t artifact_params;
    sd_vid_gen_artifact_params_init(&artifact_params);
    artifact_params.conditioning_input_path  = job.vid_gen.conditioning_input_path.empty()
                                                   ? nullptr
                                                   : job.vid_gen.conditioning_input_path.c_str();
    artifact_params.conditioning_output_path = job.vid_gen.conditioning_output_path.empty()
                                                   ? nullptr
                                                   : job.vid_gen.conditioning_output_path.c_str();
    artifact_params.latent_input_path        = job.vid_gen.latent_input_path.empty()
                                                   ? nullptr
                                                   : job.vid_gen.latent_input_path.c_str();
    artifact_params.latent_output_path       = job.vid_gen.latent_output_path.empty()
                                                   ? nullptr
                                                   : job.vid_gen.latent_output_path.c_str();

    SDImageVec results;
    int num_results                        = 0;
    sd_audio_t* generated_audio            = nullptr;
    bool attention_sparsity_restore_failed = false;
    bool artifact_only_ok                  = true;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        float previous_attention_sparsity = 0.0f;
        bool restore_attention_sparsity   = false;
        if (job.vid_gen.has_minimax_h3_attention_sparsity) {
            if (!sd_ctx_get_attention_sparsity(runtime.sd_ctx, &previous_attention_sparsity) ||
                !sd_ctx_set_attention_sparsity(runtime.sd_ctx,
                                               job.vid_gen.minimax_h3_attention_sparsity)) {
                error_message = "unable to apply attention sparsity override";
                return false;
            }
            restore_attention_sparsity = true;
            LOG_INFO("job %s: applying MiniMax-H3 attention sparsity %.1f%% (previous %.1f%%)",
                     job.id.c_str(),
                     job.vid_gen.minimax_h3_attention_sparsity * 100.0f,
                     previous_attention_sparsity * 100.0f);
        }

        sd_image_t* raw_results = nullptr;
        if (job.vid_gen.conditioning_only) {
            artifact_only_ok = encode_video_conditioning(runtime.sd_ctx,
                                                         &params,
                                                         artifact_params.conditioning_output_path);
        } else if (job.vid_gen.latent_only) {
            artifact_only_ok = sample_video_latent(runtime.sd_ctx,
                                                   &params,
                                                   artifact_params.conditioning_input_path,
                                                   artifact_params.latent_output_path);
        } else if (!generate_video_with_artifacts(runtime.sd_ctx,
                                                  &params,
                                                  &artifact_params,
                                                  &raw_results,
                                                  &num_results,
                                                  &generated_audio)) {
            raw_results = nullptr;
        }
        results.adopt(raw_results, num_results);

        if (restore_attention_sparsity) {
            attention_sparsity_restore_failed = !sd_ctx_set_attention_sparsity(runtime.sd_ctx,
                                                                               previous_attention_sparsity);
            if (!attention_sparsity_restore_failed) {
                LOG_INFO("job %s: restored MiniMax-H3 attention sparsity to %.1f%%",
                         job.id.c_str(),
                         previous_attention_sparsity * 100.0f);
            }
        }
    }

    if (attention_sparsity_restore_failed) {
        free_sd_audio(generated_audio);
        error_message = "unable to restore attention sparsity after generation";
        return false;
    }
    if (job.vid_gen.conditioning_only || job.vid_gen.latent_only) {
        if (!artifact_only_ok) {
            error_message = job.vid_gen.conditioning_only ? "conditioning encoding failed"
                                                          : "latent sampling failed";
        }
        return artifact_only_ok;
    }

    num_results = results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        error_message = "generate_video returned no results";
        return false;
    }

    std::vector<uint8_t> video_bytes = create_video_from_sd_images_to_vector(job.vid_gen.output_format,
                                                                             results.data(),
                                                                             num_results,
                                                                             job.vid_gen.gen_params.fps,
                                                                             job.vid_gen.output_compression,
                                                                             generated_audio);
    free_sd_audio(generated_audio);
    if (video_bytes.empty()) {
        error_message = "failed to encode generated video container";
        return false;
    }

    if (runtime.svr_params->output_dir.empty()) {
        output_media_b64 = base64_encode(video_bytes);
    } else if (!persist_video_output(runtime.svr_params->output_dir,
                                     job.id,
                                     job.vid_gen.output_format,
                                     video_bytes,
                                     output_media_path,
                                     output_media_file_name,
                                     error_message)) {
        return false;
    }
    output_media_mime_type = video_mime_type(job.vid_gen.output_format);
    output_frame_count     = num_results;
    output_fps             = job.vid_gen.gen_params.fps;
    return true;
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            manager.cv.wait(lock, [&]() { return manager.stop || !manager.queue.empty(); });

            if (manager.stop && manager.queue.empty()) {
                break;
            }

            purge_expired_jobs(manager);
            if (manager.queue.empty()) {
                continue;
            }

            const std::string job_id = manager.queue.front();
            manager.queue.pop_front();

            auto it = manager.jobs.find(job_id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job             = it->second;
            job->status     = AsyncJobStatus::Generating;
            job->started_at = unix_timestamp_now();
        }

        std::vector<std::string> output_images;
        std::string output_media_b64;
        std::string output_media_path;
        std::string output_media_file_name;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        std::string error_message;
        bool ok = false;

        if (job->kind == AsyncJobKind::ImgGen) {
            ok = execute_img_gen_job(runtime, *job, output_images, error_message);
        } else if (job->kind == AsyncJobKind::VidGen) {
            ok = execute_vid_gen_job(runtime,
                                     *job,
                                     output_media_b64,
                                     output_media_path,
                                     output_media_file_name,
                                     output_media_mime_type,
                                     output_frame_count,
                                     output_fps,
                                     error_message);
        } else {
            error_message = "unsupported job kind";
        }

        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto it = manager.jobs.find(job->id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job->completed_at = unix_timestamp_now();
            if (ok) {
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_path      = std::move(output_media_path);
                job->result_media_file_name = std::move(output_media_file_name);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
            } else {
                job->status        = AsyncJobStatus::Failed;
                job->error_code    = "generation_failed";
                job->error_message = error_message.empty() ? "unknown generation error" : error_message;
                job->result_images_b64.clear();
                job->result_media_b64.clear();
                job->result_media_path.clear();
                job->result_media_file_name.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            }

            purge_expired_jobs(manager);
        }
    }
}
