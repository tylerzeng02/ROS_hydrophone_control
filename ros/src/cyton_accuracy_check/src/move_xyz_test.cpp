// move_xyz_test: like move_x_test.cpp, but commands a Cartesian delta on
// all three axes at once (dx, dy, dz all plugged in on the command line),
// rather than one axis at a time. Otherwise identical mechanism: relative
// move (current pose + delta) via real IK (MoveGroupInterface::
// setPoseTarget()), verified against the NDI Polaris tracker's own
// before/after measurement of the moving marker relative to the fixed
// marker. See move_x_test.cpp for why this sidesteps the base-frame and
// tool-frame fitting problem entirely.
//
// NdiTracker and its dependencies are copied verbatim from move_x_test.cpp
// (itself copied verbatim from cyton_ndi_capture/src/ndi_measure.cpp).
// This is already hardware-validated NDI connect/BX-polling/averaging
// code, not reimplemented here.

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "moveit_msgs/msg/move_it_error_codes.hpp"

#include "ndicapi.h"

namespace {

// Configuration
constexpr const char* DEFAULT_NDI_DEVICE = "/dev/ttyUSB1";
constexpr const char* DEFAULT_MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";
constexpr const char* DEFAULT_FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";

constexpr const char* PLANNING_GROUP = "arm";
// Same values, same reasoning, as move_x_test.cpp. See that file's own
// comment for the full explanation (elbow_yaw's narrow locked window and
// why a larger shared planning-time budget addresses it more directly
// than more planning attempts alone).
constexpr double PLANNING_TIME_SECONDS = 30.0;
constexpr unsigned int NUM_PLANNING_ATTEMPTS = 15;

// Default commanded displacement (meters), in the MoveGroup's own
// planning/reference frame (normally base_link), not the NDI tracker's
// frame. Override via argv (dx, dy, dz).
constexpr double DEFAULT_DX_M = 0.0;
constexpr double DEFAULT_DY_M = 0.0;
constexpr double DEFAULT_DZ_M = 0.05;

constexpr int SETTLE_MS = 750;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// From move_x_test.cpp / cyton_ndi_capture/src/ndi_measure.cpp (NdiTracker
// and dependencies), copied verbatim.

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

        enableTool(movingToolHandle_, "moving");
        enableTool(fixedToolHandle_, "fixed");

        ndiTSTART(tracker_);
        requireNoNdiError("TSTART");
        tracking_ = true;

        std::cout << "NDI tracking started.\n";
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

double distanceMm(const NdiPoseSample& a, const NdiPoseSample& b) {
    const double dx = a.txMm - b.txMm;
    const double dy = a.tyMm - b.tyMm;
    const double dz = a.tzMm - b.tzMm;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [dx_m] [dy_m] [dz_m] [ndi_device] [moving_rom_path] [fixed_rom_path]\n"
              << "  Examples:\n"
              << "    " << argv0 << " 0.05 0 0.02      move 5cm +X and 2cm +Z simultaneously\n"
              << "    " << argv0 << " -0.03 0.01 0     move 3cm -X and 1cm +Y simultaneously\n"
              << "  Defaults:\n"
              << "    dx_m, dy_m, dz_m: " << DEFAULT_DX_M << ", " << DEFAULT_DY_M << ", "
              << DEFAULT_DZ_M << " (in the MoveGroup's planning frame -- normally base_link, "
                 "not the NDI frame)\n"
              << "    ndi_device:       " << DEFAULT_NDI_DEVICE << '\n'
              << "    moving_rom_path:  " << DEFAULT_MOVING_TOOL_ROM << '\n'
              << "    fixed_rom_path:   " << DEFAULT_FIXED_TOOL_ROM << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    const double dxM = argc > 1 ? std::stod(argv[1]) : DEFAULT_DX_M;
    const double dyM = argc > 2 ? std::stod(argv[2]) : DEFAULT_DY_M;
    const double dzM = argc > 3 ? std::stod(argv[3]) : DEFAULT_DZ_M;
    const char* ndiDevice = argc > 4 ? argv[4] : DEFAULT_NDI_DEVICE;
    const char* movingRom = argc > 5 ? argv[5] : DEFAULT_MOVING_TOOL_ROM;
    const char* fixedRom = argc > 6 ? argv[6] : DEFAULT_FIXED_TOOL_ROM;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("move_xyz_test");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);
        moveGroup.setNumPlanningAttempts(NUM_PLANNING_ATTEMPTS);

        std::cout << "Connecting to NDI tracker on " << ndiDevice << "...\n";
        NdiTracker tracker(ndiDevice, movingRom, fixedRom);
        tracker.initialize();

        std::cout << "\nCapturing BEFORE measurement...\n";
        DualToolCapture before = tracker.collectBothTools();
        std::cout << "  Before (moving-relative-to-fixed): ["
                  << before.movingRelativeToFixed.txMm << ", "
                  << before.movingRelativeToFixed.tyMm << ", "
                  << before.movingRelativeToFixed.tzMm << "] mm\n";

        geometry_msgs::msg::PoseStamped currentPose = moveGroup.getCurrentPose();
        geometry_msgs::msg::PoseStamped targetPose = currentPose;
        targetPose.pose.position.x += dxM;
        targetPose.pose.position.y += dyM;
        targetPose.pose.position.z += dzM;

        std::cout << "\nPlanning frame: " << moveGroup.getPlanningFrame() << '\n'
                  << "End-effector link: " << moveGroup.getEndEffectorLink() << '\n'
                  << "Current pose (x,y,z) m: (" << currentPose.pose.position.x << ", "
                  << currentPose.pose.position.y << ", " << currentPose.pose.position.z << ")\n"
                  << "Target pose  (x,y,z) m: (" << targetPose.pose.position.x << ", "
                  << targetPose.pose.position.y << ", " << targetPose.pose.position.z
                  << ")  [commanded delta: (" << dxM << ", " << dyM << ", " << dzM << ") m]\n";

        moveGroup.setPoseTarget(targetPose);
        std::cout << "\nPlanning and executing (this exercises real IK)...\n";
        auto result = moveGroup.move();

        if (result != moveit::core::MoveItErrorCode::SUCCESS) {
            throw std::runtime_error(
                "Move FAILED, error code " + std::to_string(result.val) +
                " -- see moveit_msgs/MoveItErrorCodes for what that means."
            );
        }

        std::cout << "Move succeeded. Settling " << SETTLE_MS << "ms before capture...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(SETTLE_MS));

        std::cout << "\nCapturing AFTER measurement...\n";
        DualToolCapture after = tracker.collectBothTools();
        std::cout << "  After (moving-relative-to-fixed): ["
                  << after.movingRelativeToFixed.txMm << ", "
                  << after.movingRelativeToFixed.tyMm << ", "
                  << after.movingRelativeToFixed.tzMm << "] mm\n";

        const double dxMm = after.movingRelativeToFixed.txMm - before.movingRelativeToFixed.txMm;
        const double dyMm = after.movingRelativeToFixed.tyMm - before.movingRelativeToFixed.tyMm;
        const double dzMm = after.movingRelativeToFixed.tzMm - before.movingRelativeToFixed.tzMm;
        const double measuredDisplacementMm =
            distanceMm(before.movingRelativeToFixed, after.movingRelativeToFixed);
        const double commandedDisplacementMm =
            std::sqrt(dxM * dxM + dyM * dyM + dzM * dzM) * 1000.0;
        const double errorMm = measuredDisplacementMm - commandedDisplacementMm;

        geometry_msgs::msg::PoseStamped achievedPose = moveGroup.getCurrentPose();

        // MoveIt-frame before/after/delta, in mm. Directly comparable
        // axis-by-axis to the commanded (dxM, dyM, dzM), unlike the NDI
        // numbers above, which are in a different, unaligned frame where
        // only the magnitude is directly comparable.
        const double moveitDxMm = (achievedPose.pose.position.x - currentPose.pose.position.x) * 1000.0;
        const double moveitDyMm = (achievedPose.pose.position.y - currentPose.pose.position.y) * 1000.0;
        const double moveitDzMm = (achievedPose.pose.position.z - currentPose.pose.position.z) * 1000.0;

        std::cout << "\n=== Result ===\n"
                  << "NDI moving-marker position BEFORE (x,y,z) mm: ("
                  << before.movingRelativeToFixed.txMm << ", " << before.movingRelativeToFixed.tyMm
                  << ", " << before.movingRelativeToFixed.tzMm << ")\n"
                  << "NDI moving-marker position AFTER  (x,y,z) mm: ("
                  << after.movingRelativeToFixed.txMm << ", " << after.movingRelativeToFixed.tyMm
                  << ", " << after.movingRelativeToFixed.tzMm << ")\n"
                  << "Commanded displacement magnitude: " << commandedDisplacementMm << " mm (delta "
                  << dxM * 1000.0 << ", " << dyM * 1000.0 << ", " << dzM * 1000.0 << " mm in "
                  << moveGroup.getPlanningFrame() << ")\n"
                  << "NDI-measured displacement (dx,dy,dz) mm: (" << dxMm << ", " << dyMm << ", "
                  << dzMm << ")\n"
                  << "NDI-measured displacement magnitude:     " << measuredDisplacementMm << " mm\n"
                  << "Deviation from commanded:                " << errorMm << " mm\n"
                  << "\n"
                  << "MoveIt-frame pose BEFORE (x,y,z) m: (" << currentPose.pose.position.x << ", "
                  << currentPose.pose.position.y << ", " << currentPose.pose.position.z << ")\n"
                  << "MoveIt-frame pose AFTER  (x,y,z) m: (" << achievedPose.pose.position.x << ", "
                  << achievedPose.pose.position.y << ", " << achievedPose.pose.position.z << ")\n"
                  << "MoveIt-frame delta (dx,dy,dz) mm: (" << moveitDxMm << ", " << moveitDyMm << ", "
                  << moveitDzMm << ")  <- directly comparable axis-by-axis to the commanded delta "
                     "above, since both are in the same frame\n"
                  << "\n"
                  << "  (note: NDI-measured displacement is in the NDI tracker's own frame, not "
                     "directly comparable axis-by-axis to the commanded MoveIt-frame delta -- only "
                     "the magnitude is directly comparable, since the two frames' axes aren't "
                     "aligned.)\n";
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
