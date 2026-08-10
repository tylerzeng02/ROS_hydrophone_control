// run_accuracy_check: single combined program that both commands the arm
// via MoveIt and captures each pose's NDI measurement, in one synchronized
// loop -- move, confirm success, settle, capture, log, repeat. Same shape
// as the original tests/test_five_pose_ndi_capture.cpp, but MoveIt does the
// moving instead of raw Dynamixel ticks.
//
// Built to replace the separate cyton_pose_commander + cyton_ndi_capture
// two-terminal workflow, which was found to have a real timing race: a
// human manually alternating Enter between two terminals could (and did)
// capture a pose before the move had actually finished, corrupting the
// resulting dataset (see CLAUDE.md's kinematic-calibration section for the
// specific incident -- two captures found to be the *same* physical pose
// because a capture landed before the arm had moved). A single program
// with a synchronous move-then-capture loop can't race with itself the
// same way; there's no human-timed handoff to get wrong.
//
// This file merges, largely verbatim:
//   - cyton_pose_commander/src/pose_commander.cpp's CSV loading,
//     MoveGroupInterface joint-space commanding, and START_STATE_INVALID
//     recovery logic.
//   - cyton_ndi_capture/src/ndi_measure.cpp's NdiTracker class and its
//     direct dependencies (already hardware-validated NDI connect/BX-
//     polling/averaging code, see CLAUDE.md).
//
// Tick<->radian conversion goes through robot_calibration.cpp's
// ticksToRadians()/radiansToTicks() (compiled directly into this binary,
// see CMakeLists.txt) -- not reimplemented here.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "moveit_msgs/msg/move_it_error_codes.hpp"

#include "ndicapi.h"

#include "robot_calibration.h"

namespace {

// ---------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------
constexpr const char* DEFAULT_INPUT_CSV =
    "/home/temp/dev/cyton_setup/build/repeatability_test_8points_labeled.csv";
constexpr const char* DEFAULT_OUTPUT_CSV = "accuracy_check_results.csv";
constexpr const char* DEFAULT_NDI_DEVICE = "/dev/ttyUSB1";
constexpr const char* DEFAULT_MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";
constexpr const char* DEFAULT_FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";

constexpr const char* PLANNING_GROUP = "arm";
constexpr double PLANNING_TIME_SECONDS = 5.0;
constexpr double RECOVERY_BUFFER_RAD = 0.01;

// Pause after a confirmed-successful move, before starting the NDI
// capture, to let any residual servo vibration settle -- matches this
// project's established SETTLING_TIME_MS convention (originally 750ms in
// test_five_pose_ndi_capture.cpp). MoveGroupInterface's move() already
// blocks until the controller reports the trajectory finished, so the arm
// is already at (or very near) the target by the time this runs; this is
// just extra margin before trusting the measurement.
constexpr int SETTLE_MS = 750;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

// ---------------------------------------------------------------------
// From cyton_pose_commander/src/pose_commander.cpp
// ---------------------------------------------------------------------

struct TestPoint {
    int pointId = 0;
    std::array<int, 7> ticks{};
    std::array<double, 7> radians{};
    bool hasHistoricalDeviation = false;
    double physicalOnlyDeviationMm = 0.0;
    double gpCorrectedDeviationMm = 0.0;
};

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<TestPoint> loadTestPoints(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open input CSV: " + path);
    }

    std::vector<TestPoint> points;
    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = splitCsvLine(line);

        if (firstLine) {
            firstLine = false;
            try {
                std::stoi(fields.at(0));
            } catch (...) {
                continue;  // header, skip
            }
        }

        if (fields.size() < 8) {
            throw std::runtime_error("Row has fewer than 8 columns (point_id + 7 ticks): " + line);
        }

        TestPoint point;
        point.pointId = std::stoi(fields[0]);
        for (std::size_t i = 0; i < 7; ++i) {
            point.ticks[i] = std::stoi(fields[1 + i]);
        }
        if (fields.size() >= 10) {
            point.hasHistoricalDeviation = true;
            point.physicalOnlyDeviationMm = std::stod(fields[8]);
            point.gpCorrectedDeviationMm = std::stod(fields[9]);
        }

        for (std::size_t i = 0; i < 7; ++i) {
            if (jointCalibrations.size() <= i) {
                throw std::runtime_error("jointCalibrations has fewer than 7 entries.");
            }
            point.radians[i] = ticksToRadians(jointCalibrations[i], point.ticks[i]);
        }

        points.push_back(point);
    }

    return points;
}

std::pair<double, double> safeBoundsRadians(const JointCalibration& calibration) {
    const double a = ticksToRadians(calibration, calibration.minTick);
    const double b = ticksToRadians(calibration, calibration.maxTick);
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    return {lo + RECOVERY_BUFFER_RAD, hi - RECOVERY_BUFFER_RAD};
}

bool sendCorrectiveTrajectory(const std::array<double, 7>& radians) {
    const std::string logPath = "/tmp/run_accuracy_check_recovery.log";

    std::ostringstream cmd;
    cmd << "ros2 action send_goal /arm_controller/follow_joint_trajectory "
           "control_msgs/action/FollowJointTrajectory \"trajectory: {joint_names: [";
    for (std::size_t j = 0; j < 7; ++j) {
        cmd << JOINT_NAMES[j] << (j + 1 < 7 ? ", " : "");
    }
    cmd << "], points: [{positions: [";
    for (std::size_t j = 0; j < 7; ++j) {
        cmd << std::setprecision(10) << radians[j] << (j + 1 < 7 ? ", " : "");
    }
    cmd << "], time_from_start: {sec: 4, nanosec: 0}}]}\" > " << logPath << " 2>&1";

    std::system(cmd.str().c_str());

    std::ifstream log(logPath);
    const std::string content(
        (std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>()
    );
    return content.find("SUCCEEDED") != std::string::npos;
}

// ---------------------------------------------------------------------
// From cyton_ndi_capture/src/ndi_measure.cpp (NdiTracker and dependencies)
// ---------------------------------------------------------------------

enum class NdiToolStatus { Detected, Missing, OutOfVolume, Disabled, LowQuality };

const char* toolStatusLabel(NdiToolStatus status) {
    switch (status) {
        case NdiToolStatus::Detected:    return "DETECTED";
        case NdiToolStatus::Missing:     return "MISSING";
        case NdiToolStatus::OutOfVolume: return "OUT_OF_VOLUME";
        case NdiToolStatus::Disabled:    return "DISABLED";
        case NdiToolStatus::LowQuality:  return "LOW_QUALITY";
    }
    return "UNKNOWN";
}

void printToolStatusIfChanged(
    const char* toolName, NdiToolStatus current, NdiToolStatus& lastPrinted, bool& everPrinted
) {
    if (everPrinted && current == lastPrinted) {
        return;
    }
    std::cout << toolName << " tool: " << toolStatusLabel(current) << '\n';
    lastPrinted = current;
    everPrinted = true;
}

struct NdiPoseSample {
    double q0 = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;
    double txMm = 0.0, tyMm = 0.0, tzMm = 0.0;
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

void normalizeQuaternion(NdiPoseSample& pose) {
    const double norm = std::sqrt(
        pose.q0 * pose.q0 + pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz
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
    double q0, double qx, double qy, double qz, const std::array<double, 3>& v
) {
    const double vx = v[0], vy = v[1], vz = v[2];
    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);
    return {{
        vx + q0 * tx + (qy * tz - qz * ty),
        vy + q0 * ty + (qz * tx - qx * tz),
        vz + q0 * tz + (qx * ty - qy * tx),
    }};
}

NdiPoseSample computeMovingRelativeToFixed(
    const NdiPoseSample& movingInCamera, const NdiPoseSample& fixedInCamera
) {
    NdiPoseSample moving = movingInCamera;
    NdiPoseSample fixed = fixedInCamera;
    normalizeQuaternion(moving);
    normalizeQuaternion(fixed);

    const double fw = fixed.q0, fx = -fixed.qx, fy = -fixed.qy, fz = -fixed.qz;

    NdiPoseSample relative;
    relative.q0 = fw * moving.q0 - fx * moving.qx - fy * moving.qy - fz * moving.qz;
    relative.qx = fw * moving.qx + fx * moving.q0 + fy * moving.qz - fz * moving.qy;
    relative.qy = fw * moving.qy - fx * moving.qz + fy * moving.q0 + fz * moving.qx;
    relative.qz = fw * moving.qz + fx * moving.qy - fy * moving.qx + fz * moving.q0;

    const std::array<double, 3> cameraDifference = {{
        moving.txMm - fixed.txMm, moving.tyMm - fixed.tyMm, moving.tzMm - fixed.tzMm,
    }};
    const std::array<double, 3> fixedFrameTranslation =
        rotateVectorByQuaternion(fw, fx, fy, fz, cameraDifference);

    relative.txMm = fixedFrameTranslation[0];
    relative.tyMm = fixedFrameTranslation[1];
    relative.tzMm = fixedFrameTranslation[2];
    relative.error = std::sqrt(moving.error * moving.error + fixed.error * fixed.error);
    relative.frameNumber = moving.frameNumber;
    relative.visibleMarkerCount = (std::min)(moving.visibleMarkerCount, fixed.visibleMarkerCount);

    normalizeQuaternion(relative);
    return relative;
}

class NdiTracker {
public:
    NdiTracker(const char* device, const char* movingRomPath, const char* fixedRomPath)
        : device_(device), movingRomPath_(movingRomPath), fixedRomPath_(fixedRomPath) {}

    ~NdiTracker() { shutdown(); }

    void initialize() {
        tracker_ = ndiOpenSerial(device_);
        if (tracker_ == nullptr) {
            throw std::runtime_error(std::string("Could not open NDI device: ") + device_);
        }

        ndiTSTOP(tracker_);
        ndiINIT(tracker_);
        requireNoNdiError("INIT");

        movingToolHandle_ = allocateAndInitializeTool(movingRomPath_, "moving");
        fixedToolHandle_ = allocateAndInitializeTool(fixedRomPath_, "fixed");

        printHandleDiagnostics(movingToolHandle_, "moving");
        printHandleDiagnostics(fixedToolHandle_, "fixed");

        enableTool(movingToolHandle_, "moving");
        enableTool(fixedToolHandle_, "fixed");

        ndiTSTART(tracker_);
        requireNoNdiError("TSTART");
        tracking_ = true;

        std::cout << "NDI tracking started.\n"
                  << "Moving tool handle: 0x" << std::hex << movingToolHandle_ << '\n'
                  << "Fixed tool handle: 0x" << fixedToolHandle_ << std::dec << '\n';

        waitForBothToolsVisible();
    }

    DualToolCapture collectBothTools() {
        waitForBothToolsVisible();

        std::cout << "Collecting " << NDI_REQUIRED_VALID_SAMPLES
                  << " synchronized samples...\n";

        std::vector<NdiPoseSample> movingAccepted;
        std::vector<NdiPoseSample> fixedAccepted;
        movingAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);
        fixedAccepted.reserve(NDI_REQUIRED_VALID_SAMPLES);

        int movingInvalidCount = 0;
        int fixedInvalidCount = 0;
        int rejectedPairCount = 0;

        NdiToolStatus lastMovingStatus = NdiToolStatus::Detected;
        NdiToolStatus lastFixedStatus = NdiToolStatus::Detected;
        bool movingStatusPrinted = false;
        bool fixedStatusPrinted = false;

        constexpr int WARNING_INTERVAL_ATTEMPTS = 1000;

        for (int attempt = 0;
             static_cast<int>(movingAccepted.size()) < NDI_REQUIRED_VALID_SAMPLES;
             ++attempt) {
            requestBxUpdate();

            printToolStatusIfChanged(
                "Moving", classifyToolStatus(movingToolHandle_), lastMovingStatus, movingStatusPrinted
            );
            printToolStatusIfChanged(
                "Fixed", classifyToolStatus(fixedToolHandle_), lastFixedStatus, fixedStatusPrinted
            );

            NdiPoseSample moving, fixed;
            const bool movingValid =
                tryReadToolFromCurrentBx(movingToolHandle_, "moving", moving, movingInvalidCount, false);
            const bool fixedValid =
                tryReadToolFromCurrentBx(fixedToolHandle_, "fixed", fixed, fixedInvalidCount, false);

            if (movingValid && fixedValid) {
                movingAccepted.push_back(moving);
                fixedAccepted.push_back(fixed);
            } else {
                ++rejectedPairCount;
            }

            if (attempt > 0 && attempt % WARNING_INTERVAL_ATTEMPTS == 0) {
                std::cout << "\nWARNING: only " << movingAccepted.size() << "/"
                          << NDI_REQUIRED_VALID_SAMPLES << " valid samples collected after "
                          << (attempt * NDI_SAMPLE_INTERVAL_MS) / 1000
                          << "s (moving invalid: " << movingInvalidCount
                          << ", fixed invalid: " << fixedInvalidCount
                          << ", rejected pairs: " << rejectedPairCount
                          << "). Check both tools have a clear line of sight (Ctrl+C to give up).\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(NDI_SAMPLE_INTERVAL_MS));
        }

        DualToolCapture capture;
        capture.movingInCamera = averageSamples(movingAccepted);
        capture.fixedInCamera = averageSamples(fixedAccepted);
        capture.movingRelativeToFixed =
            computeMovingRelativeToFixed(capture.movingInCamera.pose, capture.fixedInCamera.pose);
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
    int allocateAndInitializeTool(const char* romPath, const char* toolName) {
        ndiPHRQ(tracker_, "********", "0", "1", "**", "**");
        requireNoNdiError("PHRQ");

        const int handle = ndiGetPHRQHandle(tracker_);
        if (handle <= 0) {
            throw std::runtime_error(
                std::string("NDI did not allocate the ") + toolName + " passive tool handle."
            );
        }

        if (ndiPVWRFromFile(tracker_, handle, const_cast<char*>(romPath)) != NDI_OKAY) {
            throw std::runtime_error(std::string("Could not load ") + toolName + " ROM file: " + romPath);
        }
        requireNoNdiError("PVWRFromFile");

        ndiPINIT(tracker_, handle);
        requireNoNdiError("PINIT");

        std::cout << "Loaded " << toolName << " tool ROM into handle 0x" << std::hex << handle
                  << std::dec << ".\n";
        return handle;
    }

    void printHandleDiagnostics(int handle, const char* toolName) {
        ndiPHINF(tracker_, handle, NDI_BASIC | NDI_PART_NUMBER | NDI_MARKER_TYPE | NDI_PORT_LOCATION);
        requireNoNdiError("PHINF");

        const int status = ndiGetPHINFPortStatus(tracker_);
        const int markerType = ndiGetPHINFMarkerType(tracker_);

        char toolInfo[128] = {};
        ndiGetPHINFToolInfo(tracker_, toolInfo);
        toolInfo[127] = '\0';

        std::cout << "PHINF " << toolName << " handle 0x" << std::hex << handle << std::dec
                  << " | status=0x" << std::hex << status << std::dec << " | markerType=" << markerType
                  << " | toolInfo=\"";
        for (int i = 0; i < 30 && toolInfo[i] != '\0'; ++i) {
            const unsigned char c = static_cast<unsigned char>(toolInfo[i]);
            std::cout << ((c >= 32 && c <= 126) ? toolInfo[i] : '.');
        }
        std::cout << "\"\n";
    }

    void enableTool(int handle, const char* toolName) {
        ndiPENA(tracker_, handle, NDI_DYNAMIC);
        requireNoNdiError("PENA");
        std::cout << "Enabled " << toolName << " tool handle 0x" << std::hex << handle << std::dec
                  << ".\n";
    }

    void requestBxUpdate() {
        ndiCommand(
            tracker_, "BX:%04X",
            NDI_XFORMS_AND_STATUS | NDI_ADDITIONAL_INFO | NDI_3D_MARKER_POSITIONS |
                NDI_NOT_NORMALLY_REPORTED
        );
        requireNoNdiError("BX");
    }

    NdiToolStatus classifyToolStatus(int toolHandle) {
        float transform[8] = {};
        const int result = ndiGetBXTransform(tracker_, toolHandle, transform);

        if (result == NDI_DISABLED) return NdiToolStatus::Disabled;
        if (result == NDI_MISSING) return NdiToolStatus::Missing;

        const int status = ndiGetBXPortStatus(tracker_, toolHandle);
        if ((status & NDI_OUT_OF_VOLUME) != 0) return NdiToolStatus::OutOfVolume;
        if (!std::isfinite(transform[7]) || transform[7] > MAX_NDI_ERROR) return NdiToolStatus::LowQuality;
        return NdiToolStatus::Detected;
    }

    void waitForBothToolsVisible() {
        constexpr int WARNING_INTERVAL_ATTEMPTS = 600;
        std::cout << "Waiting for both NDI tools to become visible (Ctrl+C to give up)...\n";

        NdiToolStatus lastMovingStatus = NdiToolStatus::Detected;
        NdiToolStatus lastFixedStatus = NdiToolStatus::Detected;
        bool movingStatusPrinted = false;
        bool fixedStatusPrinted = false;

        for (int attempt = 1;; ++attempt) {
            requestBxUpdate();

            printToolStatusIfChanged(
                "Moving", classifyToolStatus(movingToolHandle_), lastMovingStatus, movingStatusPrinted
            );
            printToolStatusIfChanged(
                "Fixed", classifyToolStatus(fixedToolHandle_), lastFixedStatus, fixedStatusPrinted
            );

            NdiPoseSample moving, fixed;
            int movingInvalid = 0, fixedInvalid = 0;
            const bool movingValid =
                tryReadToolFromCurrentBx(movingToolHandle_, "moving", moving, movingInvalid, false);
            const bool fixedValid =
                tryReadToolFromCurrentBx(fixedToolHandle_, "fixed", fixed, fixedInvalid, false);

            if (movingValid && fixedValid) {
                std::cout << "Both NDI tools are visible through BX.\n";
                return;
            }

            if (attempt > 0 && attempt % WARNING_INTERVAL_ATTEMPTS == 0) {
                std::cout << "\nWARNING: still waiting for both tools to become visible after "
                          << (attempt * 50) / 1000
                          << "s. Check both tools have a clear line of sight (Ctrl+C to give up).\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    bool tryReadToolFromCurrentBx(
        int toolHandle, const char* toolName, NdiPoseSample& sample, int& invalidCount, bool printErrors
    ) {
        float transform[8] = {};
        const int result = ndiGetBXTransform(tracker_, toolHandle, transform);
        const int portStatus = ndiGetBXPortStatus(tracker_, toolHandle);
        const unsigned long frame = ndiGetBXFrame(tracker_, toolHandle);

        if (result != NDI_OKAY) {
            ++invalidCount;
            if (printErrors) {
                std::cerr << "NDI " << toolName << " BX transform rejected: result=" << result
                          << ", status=0x" << std::hex << portStatus << std::dec << ", frame=" << frame
                          << ".\n";
            }
            return false;
        }

        if ((portStatus & NDI_ENABLED) == 0 || (portStatus & NDI_OUT_OF_VOLUME) != 0) {
            ++invalidCount;
            if (printErrors) {
                std::cerr << "NDI " << toolName << " BX port status rejected: status=0x" << std::hex
                          << portStatus << std::dec << ", frame=" << frame << ".\n";
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
        sample.frameNumber = frame;
        sample.visibleMarkerCount = REQUIRED_VISIBLE_MARKERS;

        const bool finite = std::isfinite(sample.q0) && std::isfinite(sample.qx) &&
                             std::isfinite(sample.qy) && std::isfinite(sample.qz) &&
                             std::isfinite(sample.txMm) && std::isfinite(sample.tyMm) &&
                             std::isfinite(sample.tzMm) && std::isfinite(sample.error);

        if (!finite || sample.error > MAX_NDI_ERROR) {
            ++invalidCount;
            if (printErrors) {
                std::cerr << "NDI " << toolName << " BX quality rejected: error=" << sample.error
                          << ", frame=" << frame << ".\n";
            }
            return false;
        }

        return true;
    }

    AveragedNdiPose averageSamples(const std::vector<NdiPoseSample>& accepted) {
        if (accepted.empty()) {
            throw std::runtime_error("Cannot average an empty NDI sample set.");
        }

        NdiPoseSample mean;
        const NdiPoseSample& reference = accepted.front();

        for (NdiPoseSample sample : accepted) {
            const double dot = sample.q0 * reference.q0 + sample.qx * reference.qx +
                                sample.qy * reference.qy + sample.qz * reference.qz;
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
        mean.visibleMarkerCount = static_cast<int>(std::lround(mean.visibleMarkerCount / count));

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
                std::string("NDI ") + operation + " failed with error code " + std::to_string(error)
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

// ---------------------------------------------------------------------
// CSV output for this combined tool
// ---------------------------------------------------------------------

void writePoseFields(std::ofstream& csv, const std::string& prefix) {
    csv << ',' << prefix << "_q0" << ',' << prefix << "_qx" << ',' << prefix << "_qy" << ','
        << prefix << "_qz" << ',' << prefix << "_tx_mm" << ',' << prefix << "_ty_mm" << ','
        << prefix << "_tz_mm" << ',' << prefix << "_error" << ',' << prefix << "_frame" << ','
        << prefix << "_visible_markers";
}

void writeCsvHeader(std::ofstream& csv) {
    csv << "point_id,timestamp_ms";
    for (std::size_t j = 0; j < 7; ++j) {
        csv << ",target_tick_" << j;
    }
    // actual_rad_{i} (not a joint-name-based column) is a hard requirement,
    // not a style choice: calibrate_kinematics.py's load_poses_from_csv()
    // reads this exact column name pattern to feed the fitting scripts --
    // see CLAUDE.md's kinematic-calibration section for the established
    // schema this matches (the same one every quick_calibration_test*.csv
    // in this project has always used).
    for (std::size_t j = 0; j < 7; ++j) {
        csv << ",actual_rad_" << j;
    }
    writePoseFields(csv, "moving_camera");
    csv << ",moving_accepted_samples";
    writePoseFields(csv, "fixed_camera");
    csv << ",fixed_accepted_samples";
    writePoseFields(csv, "moving_relative_fixed");
    csv << '\n';
}

void appendPoseFields(std::ofstream& csv, const NdiPoseSample& pose) {
    csv << ',' << pose.q0 << ',' << pose.qx << ',' << pose.qy << ',' << pose.qz << ',' << pose.txMm
        << ',' << pose.tyMm << ',' << pose.tzMm << ',' << pose.error << ',' << pose.frameNumber << ','
        << pose.visibleMarkerCount;
}

void appendCsvRow(
    std::ofstream& csv, const TestPoint& point, const std::array<double, 7>& achievedRadians,
    const DualToolCapture& ndi
) {
    const auto timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()
    )
                                  .count();
    csv << point.pointId << ',' << timestampMs;
    for (int tick : point.ticks) {
        csv << ',' << tick;
    }
    for (double radians : achievedRadians) {
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

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [input_csv] [output_csv] [ndi_device] [moving_rom_path] [fixed_rom_path]\n"
              << "  Defaults:\n"
              << "    input_csv:       " << DEFAULT_INPUT_CSV << '\n'
              << "    output_csv:      " << DEFAULT_OUTPUT_CSV << " (appended to if it exists)\n"
              << "    ndi_device:      " << DEFAULT_NDI_DEVICE << '\n'
              << "    moving_rom_path: " << DEFAULT_MOVING_TOOL_ROM << '\n'
              << "    fixed_rom_path:  " << DEFAULT_FIXED_TOOL_ROM << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    const std::string inputCsv = argc > 1 ? argv[1] : DEFAULT_INPUT_CSV;
    const std::string outputCsv = argc > 2 ? argv[2] : DEFAULT_OUTPUT_CSV;
    const char* ndiDevice = argc > 3 ? argv[3] : DEFAULT_NDI_DEVICE;
    const char* movingRom = argc > 4 ? argv[4] : DEFAULT_MOVING_TOOL_ROM;
    const char* fixedRom = argc > 5 ? argv[5] : DEFAULT_FIXED_TOOL_ROM;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("run_accuracy_check");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::cout << "Loading test points from " << inputCsv << "...\n";
        std::vector<TestPoint> points = loadTestPoints(inputCsv);
        std::cout << "Loaded " << points.size() << " test points.\n";

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);

        std::cout << "Connecting to NDI tracker on " << ndiDevice << "...\n";
        NdiTracker tracker(ndiDevice, movingRom, fixedRom);
        tracker.initialize();

        const bool fileExistedAndNonEmpty = [&]() {
            std::ifstream check(outputCsv);
            return check.good() && check.peek() != std::ifstream::traits_type::eof();
        }();
        std::ofstream csv(outputCsv, std::ios::app);
        if (!csv) {
            throw std::runtime_error("Could not open output CSV: " + outputCsv);
        }
        if (!fileExistedAndNonEmpty) {
            writeCsvHeader(csv);
        }
        std::cout << "Logging to " << outputCsv << (fileExistedAndNonEmpty ? " (appending)\n" : " (new file)\n");

        std::cout << "\nStarting automated move+capture loop for " << points.size()
                  << " poses. No per-pose Enter needed -- each pose is only captured after a "
                     "confirmed successful move. Ctrl+C to stop early.\n";

        int completedCount = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const TestPoint& point = points[i];

            std::cout << "\n=== Pose " << (i + 1) << "/" << points.size()
                      << " (point_id=" << point.pointId << ") ===\n";
            for (std::size_t j = 0; j < 7; ++j) {
                std::cout << "  " << JOINT_NAMES[j] << ": tick=" << point.ticks[j] << ", rad="
                          << point.radians[j] << '\n';
            }
            if (point.hasHistoricalDeviation) {
                std::cout << "  (historical deviation at this point: physical-only "
                          << point.physicalOnlyDeviationMm << "mm, GP-corrected "
                          << point.gpCorrectedDeviationMm << "mm)\n";
            }

            bool anyTargetRejected = false;
            for (std::size_t j = 0; j < 7; ++j) {
                if (!moveGroup.setJointValueTarget(JOINT_NAMES[j], point.radians[j])) {
                    std::cout << "  WARNING: setJointValueTarget rejected " << JOINT_NAMES[j]
                              << " = " << point.radians[j]
                              << " rad (likely outside its URDF limit) -- skipping this pose.\n";
                    anyTargetRejected = true;
                }
            }
            if (anyTargetRejected) {
                continue;
            }

            std::cout << "  Planning and executing...\n";
            auto result = moveGroup.move();

            if (result != moveit::core::MoveItErrorCode::SUCCESS &&
                result.val == moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID) {
                std::cout << "  Move FAILED with START_STATE_INVALID (error code " << result.val
                          << ") -- attempting automatic recovery...\n";

                auto currentState = moveGroup.getCurrentState(2.0);
                if (!currentState) {
                    std::cout << "  Recovery FAILED: could not get current robot state -- "
                                 "skipping this pose.\n";
                    continue;
                }

                std::array<double, 7> corrected{};
                bool anyClamped = false;
                for (std::size_t j = 0; j < 7; ++j) {
                    double value = currentState->getVariablePosition(JOINT_NAMES[j]);
                    const auto bounds = safeBoundsRadians(jointCalibrations[j]);
                    if (value < bounds.first) {
                        value = bounds.first;
                        anyClamped = true;
                    } else if (value > bounds.second) {
                        value = bounds.second;
                        anyClamped = true;
                    }
                    corrected[j] = value;
                }

                if (!anyClamped) {
                    std::cout << "  Recovery FAILED: no joint was actually found out of bounds "
                                 "-- skipping this pose.\n";
                    continue;
                }

                std::cout << "  Sending corrective trajectory directly to the controller...\n";
                if (!sendCorrectiveTrajectory(corrected)) {
                    std::cout << "  Recovery FAILED: corrective trajectory did not report "
                                 "success -- skipping this pose.\n";
                    continue;
                }

                std::cout << "  Recovery move sent. Retrying this pose's original target...\n";
                for (std::size_t j = 0; j < 7; ++j) {
                    moveGroup.setJointValueTarget(JOINT_NAMES[j], point.radians[j]);
                }
                result = moveGroup.move();
            }

            if (result != moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Move FAILED, error code " << result.val
                          << " -- skipping NDI capture for this pose, continuing to the next.\n";
                continue;
            }

            std::cout << "  Move succeeded. Settling " << SETTLE_MS << "ms before capture...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(SETTLE_MS));

            std::array<double, 7> achievedRadians{};
            auto achievedState = moveGroup.getCurrentState(2.0);
            if (achievedState) {
                for (std::size_t j = 0; j < 7; ++j) {
                    achievedRadians[j] = achievedState->getVariablePosition(JOINT_NAMES[j]);
                }
            } else {
                std::cout << "  WARNING: could not read achieved joint state, logging target "
                             "radians instead.\n";
                achievedRadians = point.radians;
            }

            std::cout << "  Capturing NDI measurement...\n";
            DualToolCapture capture = tracker.collectBothTools();

            std::cout << "  Measured moving-relative-to-fixed: ["
                      << capture.movingRelativeToFixed.txMm << ", "
                      << capture.movingRelativeToFixed.tyMm << ", "
                      << capture.movingRelativeToFixed.tzMm << "] mm, error="
                      << capture.movingRelativeToFixed.error << '\n';

            appendCsvRow(csv, point, achievedRadians, capture);
            ++completedCount;
        }

        std::cout << "\nDone. " << completedCount << "/" << points.size()
                  << " poses moved, captured, and logged.\n";
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
