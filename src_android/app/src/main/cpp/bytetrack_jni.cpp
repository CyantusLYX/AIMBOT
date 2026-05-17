#include <jni.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace {

constexpr int kInputStride = 6;
constexpr int kOutputStride = 6;
constexpr float kMinTrackIou = 0.25F;
constexpr int kMaxMissedFrames = 8;

struct Box {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct Detection {
    Box box;
    float confidence = 0.0F;
    int class_id = 0;
};

struct Track {
    int id = 0;
    Box box;
    int class_id = 0;
    float confidence = 0.0F;
    int missed_frames = 0;
};

std::mutex g_tracker_mutex;
int g_next_tracking_id = 1;
std::vector<Track> g_tracks;

void ThrowIllegalArgumentException(JNIEnv* env, const char* message) {
    jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

float Area(const Box& box) {
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

float IntersectionOverUnion(const Box& first, const Box& second) {
    const float first_right = first.x + first.width;
    const float first_bottom = first.y + first.height;
    const float second_right = second.x + second.width;
    const float second_bottom = second.y + second.height;

    const float intersection_left = std::max(first.x, second.x);
    const float intersection_top = std::max(first.y, second.y);
    const float intersection_right = std::min(first_right, second_right);
    const float intersection_bottom = std::min(first_bottom, second_bottom);
    const float intersection_width = std::max(0.0F, intersection_right - intersection_left);
    const float intersection_height = std::max(0.0F, intersection_bottom - intersection_top);
    const float intersection_area = intersection_width * intersection_height;
    const float union_area = Area(first) + Area(second) - intersection_area;

    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return intersection_area / union_area;
}

void RemoveExpiredTracks() {
    g_tracks.erase(
            std::remove_if(
                    g_tracks.begin(),
                    g_tracks.end(),
                    [](const Track& track) {
                        return track.missed_frames > kMaxMissedFrames;
                    }),
            g_tracks.end());
}

void AgeAllTracks() {
    for (Track& track : g_tracks) {
        track.missed_frames += 1;
    }
    RemoveExpiredTracks();
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_cyantus_aimbot_detection_ByteTracker_nativeReset(
        JNIEnv* /* env */,
        jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_tracker_mutex);
    g_tracks.clear();
    g_next_tracking_id = 1;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_cyantus_aimbot_detection_ByteTracker_nativeUpdate(
        JNIEnv* env,
        jobject /* thiz */,
        jfloatArray raw_detections) {
    if (raw_detections == nullptr) {
        ThrowIllegalArgumentException(env, "rawDetections must not be null.");
        return nullptr;
    }

    const jsize input_length = env->GetArrayLength(raw_detections);
    if (input_length % kInputStride != 0) {
        ThrowIllegalArgumentException(
                env,
                "rawDetections must be a multiple of 6 floats: "
                "[x, y, width, height, confidence, classId].");
        return nullptr;
    }

    std::vector<jfloat> input(input_length);
    if (input_length > 0) {
        env->GetFloatArrayRegion(raw_detections, 0, input_length, input.data());
        if (env->ExceptionCheck()) {
            return nullptr;
        }
    }

    std::lock_guard<std::mutex> lock(g_tracker_mutex);
    const jsize detection_count = input_length / kInputStride;
    const jsize output_length = detection_count * kOutputStride;
    std::vector<jfloat> output(output_length);
    if (detection_count == 0) {
        AgeAllTracks();
        return env->NewFloatArray(0);
    }

    std::vector<Detection> detections(detection_count);
    for (jsize detection_index = 0; detection_index < detection_count; ++detection_index) {
        const jsize input_offset = detection_index * kInputStride;
        detections[detection_index] = Detection{
                Box{
                        input[input_offset],
                        input[input_offset + 1],
                        input[input_offset + 2],
                        input[input_offset + 3],
                },
                input[input_offset + 4],
                static_cast<int>(input[input_offset + 5]),
        };
    }

    std::vector<int> assigned_ids(detection_count, -1);
    std::vector<bool> track_used(g_tracks.size(), false);

    for (jsize detection_index = 0; detection_index < detection_count; ++detection_index) {
        const Detection& detection = detections[detection_index];
        int best_track_index = -1;
        float best_iou = kMinTrackIou;

        for (size_t track_index = 0; track_index < g_tracks.size(); ++track_index) {
            if (track_used[track_index] || g_tracks[track_index].class_id != detection.class_id) {
                continue;
            }

            const float iou = IntersectionOverUnion(detection.box, g_tracks[track_index].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_track_index = static_cast<int>(track_index);
            }
        }

        if (best_track_index >= 0) {
            Track& track = g_tracks[best_track_index];
            track.box = detection.box;
            track.confidence = detection.confidence;
            track.missed_frames = 0;
            track_used[best_track_index] = true;
            assigned_ids[detection_index] = track.id;
        }
    }

    for (size_t track_index = 0; track_index < g_tracks.size(); ++track_index) {
        if (!track_used[track_index]) {
            g_tracks[track_index].missed_frames += 1;
        }
    }
    RemoveExpiredTracks();

    for (jsize detection_index = 0; detection_index < detection_count; ++detection_index) {
        if (assigned_ids[detection_index] >= 0) {
            continue;
        }

        const Detection& detection = detections[detection_index];
        const int tracking_id = g_next_tracking_id++;
        g_tracks.push_back(
                Track{
                        tracking_id,
                        detection.box,
                        detection.class_id,
                        detection.confidence,
                        0,
                });
        assigned_ids[detection_index] = tracking_id;
    }

    for (jsize detection_index = 0; detection_index < detection_count; ++detection_index) {
        const jsize output_offset = detection_index * kOutputStride;
        const Detection& detection = detections[detection_index];

        output[output_offset] = static_cast<jfloat>(assigned_ids[detection_index]);
        output[output_offset + 1] = detection.box.x;
        output[output_offset + 2] = detection.box.y;
        output[output_offset + 3] = detection.box.width;
        output[output_offset + 4] = detection.box.height;
        output[output_offset + 5] = detection.confidence;
    }

    jfloatArray tracked_detections = env->NewFloatArray(output_length);
    if (tracked_detections == nullptr) {
        return nullptr;
    }

    env->SetFloatArrayRegion(tracked_detections, 0, output_length, output.data());
    return tracked_detections;
}
