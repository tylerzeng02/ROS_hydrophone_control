#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dynamixel_motor.h"
#include "robot_calibration.h"
#include "ndicapi.h"

namespace {

constexpr std::size_t JOINT_COUNT = 7;
constexpr std::size_t POSE_COUNT = 5;

constexpr const char* CYTON_DEVICE = "COM4";
constexpr int CYTON_BAUD_RATE = 1000000;
constexpr float CYTON_PROTOCOL_VERSION = 1.0F;

constexpr const char* NDI_DEVICE = "COM3";

constexpr const char* MOVING_TOOL_ROM =
    R"(C:\Users\ConformalUser\Desktop\Spectra\8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom)";

constexpr const char* FIXED_TOOL_ROM =
    R"(C:\Users\ConformalUser\Desktop\Spectra\8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom)";

constexpr const char* OUTPUT_CSV = "five_pose_ndi_capture.csv";

constexpr uint16_t MOVING_SPEED = 10;
constexpr int MOTOR_TOLERANCE_TICKS = 10;
constexpr int MOVE_TIMEOUT_SECONDS = 30;
constexpr int SETTLING_TIME_MS = 750;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_MAX_ATTEMPTS = 150;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

const std::array<std::array<uint16_t, JOINT_COUNT>, POSE_COUNT> TARGET_POSES = {{
    {{2048, 2048, 2066, 2108, 2078, 2048, 2048}},
    {{2078, 2018, 2096, 2078, 2108, 2018, 2078}},
    {{2018, 2078, 2036, 2138, 2048, 2078, 2018}},
    {{2068, 2068, 2096, 2078, 2058, 2068, 2078}},
    {{2028, 2028, 2036, 2138, 2098, 2028, 2018}},
}};

struct NdiPoseSample {
    double q0 = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double txMm = 0.0;
    double tyMm = 0.0;
    double tzMm = 0.0;
    double error = 0.0;
    unsigned long frameNumber = 0;
    int visibleMarkerCount = 0;
};

struct AveragedNdiPose {
    NdiPoseSample pose;
    int acceptedSamples = 0;
};

struct DualToolCapture {
    AveragedNdiPose movingInCamera;
    AveragedNdiPose fixedInCamera;
    NdiPoseSample movingRelativeToFixed;
};

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

void disableAll(DynamixelMotor& motor, const std::vector<int>& motorIds) {
    for (int id : motorIds) {
        motor.disableTorque(id);
    }
}

std::vector<uint16_t> readActualTicks(
    DynamixelMotor& motor,
    const std::vector<int>& motorIds
) {
    std::vector<uint16_t> ticks;
    ticks.reserve(motorIds.size());

    for (int id : motorIds) {
        uint16_t position = 0;
        if (!motor.readPosition(id, position)) {
            throw std::runtime_error(
                "Failed to read motor " + std::to_string(id)
            );
        }
        if (!motor.isPositionSafe(id, position)) {
            throw std::runtime_error(
                "Motor " + std::to_string(id) +
                " is outside its calibrated safe range."
            );
        }
        ticks.push_back(position);
    }

    return ticks;
}

std::vector<double> ticksToRadiansVector(
    const std::vector<uint16_t>& ticks
) {
    if (ticks.size() != JOINT_COUNT ||
        jointCalibrations.size() < JOINT_COUNT) {
        throw std::runtime_error("Invalid joint calibration data.");
    }

    std::vector<double> radians;
    radians.reserve(JOINT_COUNT);

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        radians.push_back(
            ticksToRadians(jointCalibrations[i], ticks[i])
        );
    }

    return radians;
}

bool targetPoseIsSafe(
    const std::array<uint16_t, JOINT_COUNT>& pose
) {
    if (jointCalibrations.size() < JOINT_COUNT) {
        return false;
    }

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        const JointCalibration& joint = jointCalibrations[i];
        if (pose[i] < joint.minTick || pose[i] > joint.maxTick) {
            std::cerr
                << "Pose rejected: motor " << joint.id
                << " target " << pose[i]
                << " is outside [" << joint.minTick
                << ", " << joint.maxTick << "].\n";
            return false;
        }
    }

    return true;
}

void normalizeQuaternion(NdiPoseSample& pose) {
    const double norm = std::sqrt(
        pose.q0 * pose.q0 +
        pose.qx * pose.qx +
        pose.qy * pose.qy +
        pose.qz * pose.qz
    );

    if (norm <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("NDI quaternion is invalid.");
    }

    pose.q0 /= norm;
    pose.qx /= norm;
    pose.qy /= norm;
    pose.qz /= norm;
}

std::array<double, 3> rotateVectorByQuaternion(
    double q0,
    double qx,
    double qy,
    double qz,
    const std::array<double, 3>& v
) {
    const double vx = v[0];
    const double vy = v[1];
    const double vz = v[2];

    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);

    return {{
        vx + q0 * tx + (qy * tz - qz * ty),
        vy + q0 * ty + (qz * tx - qx * tz),
        vz + q0 * tz + (qx * ty - qy * tx)
    }};
}

NdiPoseSample computeMovingRelativeToFixed(
    const NdiPoseSample& movingInCamera,
    const NdiPoseSample& fixedInCamera
) {
    NdiPoseSample moving = movingInCamera;
    NdiPoseSample fixed = fixedInCamera;
    normalizeQuaternion(moving);
    normalizeQuaternion(fixed);

    // q_fixed_inverse = conjugate(q_fixed), because q_fixed is unit length.
    const double fw = fixed.q0;
    const double fx = -fixed.qx;
    const double fy = -fixed.qy;
    const double fz = -fixed.qz;

    NdiPoseSample relative;

    // q_relative = q_fixed_inverse * q_moving
    relative.q0 =
        fw * moving.q0 - fx * moving.qx -
        fy * moving.qy - fz * moving.qz;
    relative.qx =
        fw * moving.qx + fx * moving.q0 +
        fy * moving.qz - fz * moving.qy;
    relative.qy =
        fw * moving.qy - fx * moving.qz +
        fy * moving.q0 + fz * moving.qx;
    relative.qz =
        fw * moving.qz + fx * moving.qy -
        fy * moving.qx + fz * moving.q0;

    const std::array<double, 3> cameraDifference = {{
        moving.txMm - fixed.txMm,
        moving.tyMm - fixed.tyMm,
        moving.tzMm - fixed.tzMm
    }};

    const std::array<double, 3> fixedFrameTranslation =
        rotateVectorByQuaternion(fw, fx, fy, fz, cameraDifference);

    relative.txMm = fixedFrameTranslation[0];
    relative.tyMm = fixedFrameTranslation[1];
    relative.tzMm = fixedFrameTranslation[2];

    relative.error = std::sqrt(
        moving.error * moving.error +
        fixed.error * fixed.error
    );
    relative.frameNumber = moving.frameNumber;
    relative.visibleMarkerCount = (std::min)(
        moving.visibleMarkerCount,
        fixed.visibleMarkerCount
    );

    normalizeQuaternion(relative);
    return relative;
}

class NdiTracker {
public:
    NdiTracker(
        const char* device,
        const char* movingRomPath,
        const char* fixedRomPath
    )
        : device_(device),
          movingRomPath_(movingRomPath),
          fixedRomPath_(fixedRomPath) {}

    ~NdiTracker() {
        shutdown();
    }

    void initialize() {
        tracker_ = ndiOpenSerial(device_);
        if (tracker_ == nullptr) {
            throw std::runtime_error(
                std::string("Could not open NDI device: ") + device_
            );
        }

        ndiINIT(tracker_);
        requireNoNdiError("INIT");

        movingToolHandle_ = allocateAndInitializeTool(
            movingRomPath_,
            "moving"
        );

        fixedToolHandle_ = allocateAndInitializeTool(
            fixedRomPath_,
            "fixed"
        );

        enableTool(movingToolHandle_, "moving");
        enableTool(fixedToolHandle_, "fixed");

        ndiTSTART(tracker_);
        requireNoNdiError("TSTART");
        tracking_ = true;

        std::cout
            << "NDI tracking started.\n"
            << "Moving tool handle: 0x"
            << std::hex << movingToolHandle_ << '\n'
            << "Fixed tool handle: 0x"
            << fixedToolHandle_ << std::dec << '\n';

        waitForBothToolsVisible();
    }

    DualToolCapture collectBothTools() {
        std::vector<NdiPoseSample> movingAccepted;
        std::vector<NdiPoseSample> fixedAccepted;

        movingAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);
        fixedAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);

        int movingInvalidCount = 0;
        int fixedInvalidCount = 0;
        int rejectedPairCount = 0;

        for (int attempt = 0;
             attempt < NDI_MAX_ATTEMPTS &&
             static_cast<int>(movingAccepted.size()) <
                 NDI_REQUIRED_VALID_SAMPLES;
             ++attempt) {

            // One TX command updates both tools from the same Polaris frame.
            ndiTX(
                tracker_,
                NDI_XFORMS_AND_STATUS |
                NDI_ADDITIONAL_INFO |
                NDI_FRAME_NUMBER
            );
            requireNoNdiError("TX");

            NdiPoseSample moving;
            NdiPoseSample fixed;

            const bool movingValid = tryReadToolFromCurrentTx(
                movingToolHandle_,
                "moving",
                moving,
                movingInvalidCount
            );

            const bool fixedValid = tryReadToolFromCurrentTx(
                fixedToolHandle_,
                "fixed",
                fixed,
                fixedInvalidCount
            );

            if (movingValid && fixedValid) {
                movingAccepted.push_back(moving);
                fixedAccepted.push_back(fixed);
            } else {
                ++rejectedPairCount;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS)
            );
        }

        if (static_cast<int>(movingAccepted.size()) <
            NDI_REQUIRED_VALID_SAMPLES) {
            throw std::runtime_error(
                "Not enough synchronized NDI samples. Accepted " +
                std::to_string(movingAccepted.size()) + " of " +
                std::to_string(NDI_REQUIRED_VALID_SAMPLES) +
                ". Moving missing/invalid: " +
                std::to_string(movingInvalidCount) +
                ", fixed missing/invalid: " +
                std::to_string(fixedInvalidCount) +
                ", rejected pairs: " +
                std::to_string(rejectedPairCount) + "."
            );
        }

        DualToolCapture capture;
        capture.movingInCamera = averageSamples(movingAccepted);
        capture.fixedInCamera = averageSamples(fixedAccepted);
        capture.movingRelativeToFixed = computeMovingRelativeToFixed(
            capture.movingInCamera.pose,
            capture.fixedInCamera.pose
        );

        return capture;
    }

    void shutdown() noexcept {
        if (tracker_ == nullptr) {
            return;
        }

        if (tracking_) {
            ndiTSTOP(tracker_);
            tracking_ = false;
        }

        releaseTool(movingToolHandle_);
        releaseTool(fixedToolHandle_);

        ndiCloseSerial(tracker_);
        tracker_ = nullptr;
    }

private:
    int allocateAndInitializeTool(
        const char* romPath,
        const char* toolName
    ) {
        ndiPHRQ(tracker_, "********", "0", "1", "**", "**");
        requireNoNdiError("PHRQ");

        const int handle = ndiGetPHRQHandle(tracker_);
        if (handle <= 0) {
            throw std::runtime_error(
                std::string("NDI did not allocate the ") +
                toolName + " passive tool handle."
            );
        }

        if (ndiPVWRFromFile(
                tracker_,
                handle,
                const_cast<char*>(romPath)
            ) != NDI_OKAY) {
            throw std::runtime_error(
                std::string("Could not load ") + toolName +
                " ROM file: " + romPath
            );
        }
        requireNoNdiError("PVWRFromFile");

        ndiPINIT(tracker_, handle);
        requireNoNdiError("PINIT");

        std::cout
            << "Loaded " << toolName
            << " tool ROM into handle 0x"
            << std::hex << handle << std::dec << ".\n";

        return handle;
    }

    void enableTool(int handle, const char* toolName) {
        ndiPENA(tracker_, handle, NDI_DYNAMIC);
        requireNoNdiError("PENA");

        std::cout
            << "Enabled " << toolName
            << " tool handle 0x"
            << std::hex << handle << std::dec << ".\n";
    }

    void waitForBothToolsVisible() {
        constexpr int STARTUP_ATTEMPTS = 100;

        for (int attempt = 1; attempt <= STARTUP_ATTEMPTS; ++attempt) {
            ndiTX(
                tracker_,
                NDI_XFORMS_AND_STATUS |
                NDI_ADDITIONAL_INFO |
                NDI_FRAME_NUMBER
            );
            requireNoNdiError("startup TX");

            NdiPoseSample moving;
            NdiPoseSample fixed;
            int movingInvalid = 0;
            int fixedInvalid = 0;

            const bool movingValid = tryReadToolFromCurrentTx(
                movingToolHandle_,
                "moving",
                moving,
                movingInvalid,
                false
            );

            const bool fixedValid = tryReadToolFromCurrentTx(
                fixedToolHandle_,
                "fixed",
                fixed,
                fixedInvalid,
                false
            );

            if (movingValid && fixedValid) {
                std::cout
                    << "Both NDI tools are visible and producing transforms.\n";
                return;
            }

            if (attempt == 1 || attempt % 20 == 0) {
                std::cout
                    << "Waiting for both NDI tools..."
                    << " moving=" << (movingValid ? "valid" : "invalid")
                    << ", fixed=" << (fixedValid ? "valid" : "invalid")
                    << '\n';
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(50)
            );
        }

        throw std::runtime_error(
            "Both NDI tools were not visible after startup. "
            "Verify that rigid body 2 uses the moving ROM, rigid body 3 "
            "uses the fixed ROM, all four spheres are visible, and "
            "the NDI tracking application is closed."
        );
    }

    bool tryReadToolFromCurrentTx(
        int toolHandle,
        const char* toolName,
        NdiPoseSample& sample,
        int& invalidCount,
        bool printErrors = true
    ) {
        double transform[8] = {};
        const int result = ndiGetTXTransform(
            tracker_,
            toolHandle,
            transform
        );

        if (result != NDI_OKAY) {
            ++invalidCount;

            if (printErrors) {
                if (result == NDI_MISSING) {
                    std::cerr
                        << "NDI " << toolName
                        << " sample skipped: marker is missing.\n";
                } else if (result == NDI_DISABLED) {
                    std::cerr
                        << "NDI " << toolName
                        << " sample skipped: tool is disabled.\n";
                } else {
                    std::cerr
                        << "NDI " << toolName
                        << " sample skipped: transform result code "
                        << result << ".\n";
                }
            }
            return false;
        }

        const int portStatus = ndiGetTXPortStatus(
            tracker_,
            toolHandle
        );

        if ((portStatus & NDI_ENABLED) == 0 ||
            (portStatus & NDI_OUT_OF_VOLUME) != 0) {
            ++invalidCount;

            if (printErrors) {
                std::cerr
                    << "NDI " << toolName
                    << " sample skipped: invalid port status "
                    << portStatus << ".\n";
            }
            return false;
        }

        sample.q0 = transform[0];
        sample.qx = transform[1];
        sample.qy = transform[2];
        sample.qz = transform[3];
        sample.txMm = transform[4];
        sample.tyMm = transform[5];
        sample.tzMm = transform[6];
        sample.error = transform[7];
        sample.frameNumber = ndiGetTXFrame(tracker_, toolHandle);

        int visibleMarkers = 0;
        for (int marker = 0; marker < 4; ++marker) {
            if (ndiGetTXMarkerInfo(
                    tracker_,
                    toolHandle,
                    marker
                ) == NDI_MARKER_USED) {
                ++visibleMarkers;
            }
        }
        sample.visibleMarkerCount = visibleMarkers;

        const bool finite =
            std::isfinite(sample.q0) &&
            std::isfinite(sample.qx) &&
            std::isfinite(sample.qy) &&
            std::isfinite(sample.qz) &&
            std::isfinite(sample.txMm) &&
            std::isfinite(sample.tyMm) &&
            std::isfinite(sample.tzMm) &&
            std::isfinite(sample.error);

        const bool qualityAccepted =
            finite &&
            sample.error <= MAX_NDI_ERROR;

        if (!qualityAccepted) {
            ++invalidCount;

            if (printErrors) {
                std::cerr
                    << "NDI " << toolName
                    << " sample rejected: markers="
                    << sample.visibleMarkerCount
                    << ", error=" << sample.error << ".\n";
            }
            return false;
        }

        return true;
    }

    AveragedNdiPose averageSamples(
        const std::vector<NdiPoseSample>& accepted
    ) {
        if (accepted.empty()) {
            throw std::runtime_error(
                "Cannot average an empty NDI sample set."
            );
        }

        NdiPoseSample mean;
        const NdiPoseSample& reference = accepted.front();

        for (NdiPoseSample sample : accepted) {
            const double dot =
                sample.q0 * reference.q0 +
                sample.qx * reference.qx +
                sample.qy * reference.qy +
                sample.qz * reference.qz;

            if (dot < 0.0) {
                sample.q0 = -sample.q0;
                sample.qx = -sample.qx;
                sample.qy = -sample.qy;
                sample.qz = -sample.qz;
            }

            mean.q0 += sample.q0;
            mean.qx += sample.qx;
            mean.qy += sample.qy;
            mean.qz += sample.qz;
            mean.txMm += sample.txMm;
            mean.tyMm += sample.tyMm;
            mean.tzMm += sample.tzMm;
            mean.error += sample.error;
            mean.frameNumber = sample.frameNumber;
            mean.visibleMarkerCount += sample.visibleMarkerCount;
        }

        const double count = static_cast<double>(accepted.size());
        mean.q0 /= count;
        mean.qx /= count;
        mean.qy /= count;
        mean.qz /= count;
        mean.txMm /= count;
        mean.tyMm /= count;
        mean.tzMm /= count;
        mean.error /= count;
        mean.visibleMarkerCount = static_cast<int>(
            std::lround(mean.visibleMarkerCount / count)
        );

        normalizeQuaternion(mean);

        return {mean, static_cast<int>(accepted.size())};
    }

    void releaseTool(int& handle) noexcept {
        if (handle > 0) {
            ndiPDIS(tracker_, handle);
            ndiPHF(tracker_, handle);
            handle = 0;
        }
    }

    void requireNoNdiError(const char* operation) {
        const int error = ndiGetError(tracker_);
        if (error != NDI_OKAY) {
            throw std::runtime_error(
                std::string("NDI ") + operation +
                " failed with error code " +
                std::to_string(error)
            );
        }
    }

    const char* device_;
    const char* movingRomPath_;
    const char* fixedRomPath_;
    ndicapi* tracker_ = nullptr;
    int movingToolHandle_ = 0;
    int fixedToolHandle_ = 0;
    bool tracking_ = false;
};

void writePoseFields(
    std::ofstream& csv,
    const std::string& prefix
) {
    csv
        << ',' << prefix << "_q0"
        << ',' << prefix << "_qx"
        << ',' << prefix << "_qy"
        << ',' << prefix << "_qz"
        << ',' << prefix << "_tx_mm"
        << ',' << prefix << "_ty_mm"
        << ',' << prefix << "_tz_mm"
        << ',' << prefix << "_error"
        << ',' << prefix << "_frame"
        << ',' << prefix << "_visible_markers";
}

void writeCsvHeader(std::ofstream& csv) {
    csv << "pose_id,timestamp_ms";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",target_tick_" << i;
    }
    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",actual_tick_" << i;
    }
    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        csv << ",actual_rad_" << i;
    }

    writePoseFields(csv, "moving_camera");
    csv << ",moving_accepted_samples";

    writePoseFields(csv, "fixed_camera");
    csv << ",fixed_accepted_samples";

    writePoseFields(csv, "moving_relative_fixed");
    csv << '\n';
}

void appendPoseFields(
    std::ofstream& csv,
    const NdiPoseSample& pose
) {
    csv
        << ',' << pose.q0
        << ',' << pose.qx
        << ',' << pose.qy
        << ',' << pose.qz
        << ',' << pose.txMm
        << ',' << pose.tyMm
        << ',' << pose.tzMm
        << ',' << pose.error
        << ',' << pose.frameNumber
        << ',' << pose.visibleMarkerCount;
}

void appendCsvRow(
    std::ofstream& csv,
    std::size_t poseIndex,
    const std::array<uint16_t, JOINT_COUNT>& targetTicks,
    const std::vector<uint16_t>& actualTicks,
    const std::vector<double>& actualRadians,
    const DualToolCapture& ndi
) {
    const auto timestampMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

    csv << poseIndex + 1 << ',' << timestampMs;

    for (uint16_t tick : targetTicks) {
        csv << ',' << tick;
    }
    for (uint16_t tick : actualTicks) {
        csv << ',' << tick;
    }
    for (double radians : actualRadians) {
        csv << ',' << std::setprecision(12) << radians;
    }

    appendPoseFields(csv, ndi.movingInCamera.pose);
    csv << ',' << ndi.movingInCamera.acceptedSamples;

    appendPoseFields(csv, ndi.fixedInCamera.pose);
    csv << ',' << ndi.fixedInCamera.acceptedSamples;

    appendPoseFields(csv, ndi.movingRelativeToFixed);
    csv << '\n';

    csv.flush();
}

void printPose(
    const std::string& label,
    const NdiPoseSample& pose
) {
    std::cout
        << label << " translation: ["
        << pose.txMm << ", "
        << pose.tyMm << ", "
        << pose.tzMm << "] mm\n"
        << label << " quaternion: ["
        << pose.q0 << ", "
        << pose.qx << ", "
        << pose.qy << ", "
        << pose.qz << "]\n"
        << label << " error: " << pose.error << '\n'
        << label << " visible markers: "
        << pose.visibleMarkerCount << '\n';
}

void printCapturedPose(
    std::size_t poseIndex,
    const std::vector<uint16_t>& actualTicks,
    const std::vector<double>& actualRadians,
    const DualToolCapture& ndi
) {
    std::cout << "\nCaptured pose " << poseIndex + 1 << ":\n";

    for (std::size_t i = 0; i < JOINT_COUNT; ++i) {
        std::cout
            << "Motor " << i
            << " | " << actualTicks[i] << " ticks"
            << " | " << std::fixed << std::setprecision(6)
            << actualRadians[i] << " rad\n";
    }

    printPose("Moving tool in camera frame", ndi.movingInCamera.pose);
    std::cout
        << "Moving accepted samples: "
        << ndi.movingInCamera.acceptedSamples << '\n';

    printPose("Fixed tool in camera frame", ndi.fixedInCamera.pose);
    std::cout
        << "Fixed accepted samples: "
        << ndi.fixedInCamera.acceptedSamples << '\n';

    printPose(
        "Moving tool relative to fixed tool",
        ndi.movingRelativeToFixed
    );
}

}  // namespace

int main() {
    const std::vector<int> motorIds = {0, 1, 2, 3, 4, 5, 6};

    DynamixelMotor motor(
        CYTON_DEVICE,
        CYTON_BAUD_RATE,
        CYTON_PROTOCOL_VERSION
    );

    try {
        for (const auto& pose : TARGET_POSES) {
            if (!targetPoseIsSafe(pose)) {
                throw std::runtime_error(
                    "At least one predefined target pose is unsafe."
                );
            }
        }

        std::ofstream csv(
            OUTPUT_CSV,
            std::ios::out | std::ios::trunc
        );
        if (!csv) {
            throw std::runtime_error(
                std::string("Could not open CSV: ") + OUTPUT_CSV
            );
        }
        writeCsvHeader(csv);

        if (!motor.connect()) {
            throw std::runtime_error(
                "Could not connect to the Cyton motors."
            );
        }

        for (int id : motorIds) {
            if (!motor.pingMotor(id)) {
                throw std::runtime_error(
                    "Could not ping motor " + std::to_string(id)
                );
            }
        }

        NdiTracker ndi(
            NDI_DEVICE,
            MOVING_TOOL_ROM,
            FIXED_TOOL_ROM
        );
        ndi.initialize();

        std::cout
            << "\nFive-pose Cyton + dual-tool NDI capture test\n"
            << "Output: " << OUTPUT_CSV << "\n"
            << "Moving marker: rigid body 2\n"
            << "Fixed marker: rigid body 3\n"
            << "Each Enter press commands exactly one pose.\n"
            << "Press Ctrl+C or use your hardware emergency stop "
            << "if needed.\n";

        for (std::size_t poseIndex = 0;
             poseIndex < TARGET_POSES.size();
             ++poseIndex) {
            waitForEnter(
                "\nPress Enter to move to pose " +
                std::to_string(poseIndex + 1) + " of " +
                std::to_string(POSE_COUNT) + "..."
            );

            const auto& targetArray = TARGET_POSES[poseIndex];
            const std::vector<uint16_t> targetTicks(
                targetArray.begin(),
                targetArray.end()
            );

            if (!motor.moveJointsSafely(
                    motorIds,
                    targetTicks,
                    MOVING_SPEED,
                    MOTOR_TOLERANCE_TICKS,
                    MOVE_TIMEOUT_SECONDS,
                    true
                )) {
                throw std::runtime_error(
                    "Robot movement failed at pose " +
                    std::to_string(poseIndex + 1)
                );
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(SETTLING_TIME_MS)
            );

            const std::vector<uint16_t> actualTicks =
                readActualTicks(motor, motorIds);

            const std::vector<double> actualRadians =
                ticksToRadiansVector(actualTicks);

            const DualToolCapture ndiCapture =
                ndi.collectBothTools();

            appendCsvRow(
                csv,
                poseIndex,
                targetArray,
                actualTicks,
                actualRadians,
                ndiCapture
            );

            printCapturedPose(
                poseIndex,
                actualTicks,
                actualRadians,
                ndiCapture
            );
        }

        std::cout
            << "\nAll five poses were captured successfully.\n"
            << "Saved data to " << OUTPUT_CSV << '\n';

        disableAll(motor, motorIds);
        motor.disconnect();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        disableAll(motor, motorIds);
        motor.disconnect();
        return 1;
    }
}