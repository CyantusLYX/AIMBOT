#include <jni.h>

#include <atomic>
#include <vector>

namespace {

constexpr int kInputStride = 6;
constexpr int kOutputStride = 6;

std::atomic<int> g_next_tracking_id{1};

void ThrowIllegalArgumentException(JNIEnv* env, const char* message) {
    jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

}  // namespace

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
    env->GetFloatArrayRegion(raw_detections, 0, input_length, input.data());
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    const jsize detection_count = input_length / kInputStride;
    const jsize output_length = detection_count * kOutputStride;
    std::vector<jfloat> output(output_length);

    for (jsize detection_index = 0; detection_index < detection_count; ++detection_index) {
        const jsize input_offset = detection_index * kInputStride;
        const jsize output_offset = detection_index * kOutputStride;
        const int tracking_id = g_next_tracking_id.fetch_add(1);

        output[output_offset] = static_cast<jfloat>(tracking_id);
        output[output_offset + 1] = input[input_offset];
        output[output_offset + 2] = input[input_offset + 1];
        output[output_offset + 3] = input[input_offset + 2];
        output[output_offset + 4] = input[input_offset + 3];
        output[output_offset + 5] = input[input_offset + 4];
    }

    jfloatArray tracked_detections = env->NewFloatArray(output_length);
    if (tracked_detections == nullptr) {
        return nullptr;
    }

    env->SetFloatArrayRegion(tracked_detections, 0, output_length, output.data());
    return tracked_detections;
}
