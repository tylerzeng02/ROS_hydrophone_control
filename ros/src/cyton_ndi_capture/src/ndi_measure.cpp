// ndi_measure: standalone NDI Polaris measurement tool for checking
// MoveIt-commanded poses against independent NDI ground truth.
//
// Workflow: position the arm however you like via MoveIt/RViz (this tool
// never commands the arm itself), then press Enter here to record one
// measurement -- it logs the live /joint_states and MoveGroupInterface's
// own live getCurrentPose() (base_link frame) alongside the NDI-measured
// moving-tool-relative-to-fixed-tool pose to CSV. Repeat for as many poses
// as you want; Ctrl+C to quit.
//
// This is the NDI *measurement* half of
// calibration/collection/ndi_capture_and_validate.cpp only -- the
// NdiTracker class and its direct dependencies (structs, quaternion math,
// BX polling/averaging) are ported over close to verbatim, since that code
// is already hardware-validated. Real changes from the original:
//   1. The Windows-only <conio.h> pause/skip/manual-mode hotkeys are
//      stripped entirely (handleUserControls() is now a no-op) -- this tool
//      doesn't drive the arm through a long unattended pose list the way
//      the original did, so there's nothing long-running to pause or skip.
//      If a tool loses tracking visibility, this just keeps retrying
//      indefinitely (same "warn periodically, never abort" philosophy as
//      the original) until you either fix visibility or Ctrl+C.
//   2. Joint values come from live ROS /joint_states (whatever MoveIt/
//      ros2_control last reported), not from directly reading Dynamixel
//      ticks -- this tool has no DynamixelMotor dependency at all.
//   3. Each capture also logs MoveGroupInterface::getCurrentPose(): pairing
//      that MoveIt-frame pose with the same capture's NDI-frame pose is
//      exactly the input a Kabsch/Procrustes fit needs to re-derive
//      move_between_points.cpp's hardcoded NDI-to-MoveIt rotation, since
//      the fixed marker's orientation can drift between sessions. Using
//      MoveIt's own live pose here, rather than recomputing FK offline in
//      Python from the joint angles, means this is exactly what MoveIt
//      itself believes the pose is -- no risk of a subtle mismatch between
//      an offline reimplementation and whatever the real URDF/robot_state
//      is doing internally.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"

#include "ndicapi.h"

namespace {

// Configuration -- machine-specific, expect to update these per machine.
// Override via command-line args if needed; see printUsage() below.
constexpr const char* DEFAULT_NDI_DEVICE = "/dev/ttyUSB1";
constexpr const char* DEFAULT_MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";
constexpr const char* DEFAULT_FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";
constexpr const char* DEFAULT_OUTPUT_CSV = "moveit_ndi_accuracy_check.csv";

// Same planning group as move_between_points.cpp/run_accuracy_check.cpp.
constexpr const char* PLANNING_GROUP = "arm";

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// Joint order matches robot_calibration.cpp's jointCalibrations / this
// workspace's <ros2_control> block (motor-ID order, gripper excluded).
constexpr std::array<const char*, 7> JOINT_NAMES = {
    "shoulder_roll_joint", "shoulder_pitch_joint", "shoulder_yaw_joint",
    "elbow_pitch_joint",   "elbow_yaw_joint",       "wrist_pitch_joint",
    "wrist_roll_joint",
};

// Ported near-verbatim from ndi_capture_and_validate.cpp -- see that
// file for the full derivation history of these types/functions.

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

// Stripped down from the original's handleUserControls(): no <conio.h>,
// no pause/skip/manual-mode hotkeys, no PoseSkippedByUser exception. Kept
// as a named function (rather than deleting every call site) so the
// waitForBothToolsVisible()/collectBothTools() bodies below stay close to
// the original and are easy to diff against it.
void handleUserControls() {}

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
            handleUserControls();
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
            handleUserControls();
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

// New code: live joint state + CSV logging + the capture loop.

class JointStateCache : public rclcpp::Node {
public:
    JointStateCache() : Node("ndi_measure_joint_state_cache") {
        sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                latest_ = msg;
            }
        );
    }

    sensor_msgs::msg::JointState::SharedPtr latest() {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_;
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
    std::mutex mutex_;
    sensor_msgs::msg::JointState::SharedPtr latest_;
};

// Looks each of JOINT_NAMES up by name in the message rather than assuming
// index order matches -- sensor_msgs/JointState makes no ordering
// guarantee, and joint_state_broadcaster's default config (no explicit
// 'joints' list, see cyton_moveit_config's ros2_controllers.yaml comment)
// publishes "all available state interfaces" in whatever order the
// hardware component exported them.
bool lookUpJointRadians(
    const sensor_msgs::msg::JointState& js, std::array<double, 7>& out, std::string& missingJoint
) {
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
        auto it = std::find(js.name.begin(), js.name.end(), JOINT_NAMES[i]);
        if (it == js.name.end()) {
            missingJoint = JOINT_NAMES[i];
            return false;
        }
        out[i] = js.position[static_cast<std::size_t>(std::distance(js.name.begin(), it))];
    }
    return true;
}

void waitForEnter(const std::string& message) {
    std::cout << message << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

void writePoseFields(std::ofstream& csv, const std::string& prefix) {
    csv << ',' << prefix << "_q0" << ',' << prefix << "_qx" << ',' << prefix << "_qy" << ','
        << prefix << "_qz" << ',' << prefix << "_tx_mm" << ',' << prefix << "_ty_mm" << ','
        << prefix << "_tz_mm" << ',' << prefix << "_error" << ',' << prefix << "_frame" << ','
        << prefix << "_visible_markers";
}

// MoveGroupInterface's own live pose (base_link frame) -- the MoveIt-frame
// side of the NDI-frame/MoveIt-frame pairs a Kabsch/Procrustes fit needs to
// re-derive move_between_points.cpp's NDI-to-MoveIt rotation. Position is
// converted m -> mm to match every other length unit already in this CSV.
void writeMoveitPoseFields(std::ofstream& csv) {
    csv << ",moveit_pose_x_mm,moveit_pose_y_mm,moveit_pose_z_mm,"
           "moveit_pose_qw,moveit_pose_qx,moveit_pose_qy,moveit_pose_qz";
}

void appendMoveitPoseFields(std::ofstream& csv, const geometry_msgs::msg::PoseStamped& pose) {
    csv << ',' << std::setprecision(12) << (pose.pose.position.x * 1000.0) << ','
        << (pose.pose.position.y * 1000.0) << ',' << (pose.pose.position.z * 1000.0) << ','
        << pose.pose.orientation.w << ',' << pose.pose.orientation.x << ','
        << pose.pose.orientation.y << ',' << pose.pose.orientation.z;
}

void writeCsvHeader(std::ofstream& csv) {
    csv << "capture_id,timestamp_ms";
    for (const char* name : JOINT_NAMES) {
        csv << ',' << name << "_rad";
    }
    writeMoveitPoseFields(csv);
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
    std::ofstream& csv, int captureId, const std::array<double, 7>& jointRadians,
    const geometry_msgs::msg::PoseStamped& moveitPose, const DualToolCapture& ndi
) {
    const auto timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()
    )
                                  .count();
    csv << captureId << ',' << timestampMs;
    for (double radians : jointRadians) {
        csv << ',' << std::setprecision(12) << radians;
    }
    appendMoveitPoseFields(csv, moveitPose);
    appendPoseFields(csv, ndi.movingInCamera.pose);
    csv << ',' << ndi.movingInCamera.acceptedSamples;
    appendPoseFields(csv, ndi.fixedInCamera.pose);
    csv << ',' << ndi.fixedInCamera.acceptedSamples;
    appendPoseFields(csv, ndi.movingRelativeToFixed);
    csv << '\n';
    csv.flush();
}

void printPose(const std::string& label, const NdiPoseSample& pose) {
    std::cout << label << " translation: [" << pose.txMm << ", " << pose.tyMm << ", " << pose.tzMm
              << "] mm\n"
              << label << " error: " << pose.error << ", visible markers: "
              << pose.visibleMarkerCount << '\n';
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [ndi_device] [moving_rom_path] [fixed_rom_path] [output_csv]\n"
              << "  Defaults:\n"
              << "    ndi_device:      " << DEFAULT_NDI_DEVICE << '\n'
              << "    moving_rom_path: " << DEFAULT_MOVING_TOOL_ROM << '\n'
              << "    fixed_rom_path:  " << DEFAULT_FIXED_TOOL_ROM << '\n'
              << "    output_csv:      " << DEFAULT_OUTPUT_CSV << " (appended to if it exists)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    const char* device = argc > 1 ? argv[1] : DEFAULT_NDI_DEVICE;
    const char* movingRom = argc > 2 ? argv[2] : DEFAULT_MOVING_TOOL_ROM;
    const char* fixedRom = argc > 3 ? argv[3] : DEFAULT_FIXED_TOOL_ROM;
    const std::string outputCsv = argc > 4 ? argv[4] : DEFAULT_OUTPUT_CSV;

    // See the identical comment in ndi_status_monitor.cpp: without this,
    // std::cout is fully buffered whenever stdout isn't a TTY, and prompts/
    // status lines silently never reach the terminal/log until exit.
    std::cout.setf(std::ios::unitbuf);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointStateCache>();
    std::thread spinThread([&]() { rclcpp::spin(node); });

    int exitCode = 0;
    try {
        std::cout << "Connecting to MoveGroupInterface (group '" << PLANNING_GROUP << "')...\n";
        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);

        std::cout << "Connecting to NDI tracker on " << device << "...\n";
        NdiTracker tracker(device, movingRom, fixedRom);
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
        std::cout << "Logging to " << outputCsv
                  << (fileExistedAndNonEmpty ? " (appending)\n" : " (new file)\n");

        std::cout << "\nWaiting for /joint_states...\n";
        while (rclcpp::ok() && node->latest() == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        int captureId = 0;
        while (rclcpp::ok()) {
            waitForEnter(
                "\nPosition the arm via MoveIt, then press Enter to capture "
                "(Ctrl+C to quit)... "
            );
            if (!rclcpp::ok()) {
                break;
            }

            auto js = node->latest();
            if (!js) {
                std::cout << "No /joint_states received yet, try again.\n";
                continue;
            }

            std::array<double, 7> jointRadians{};
            std::string missingJoint;
            if (!lookUpJointRadians(*js, jointRadians, missingJoint)) {
                std::cout << "Joint '" << missingJoint
                          << "' not found in /joint_states, skipping this capture.\n";
                continue;
            }

            const geometry_msgs::msg::PoseStamped moveitPose = moveGroup.getCurrentPose();

            DualToolCapture capture = tracker.collectBothTools();
            ++captureId;

            std::cout << "\nCaptured #" << captureId << ":\n";
            for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
                std::cout << "  " << JOINT_NAMES[i] << ": " << std::fixed << std::setprecision(6)
                          << jointRadians[i] << " rad\n";
            }
            std::cout << "  MoveIt pose (base_link, mm): ["
                      << moveitPose.pose.position.x * 1000.0 << ", "
                      << moveitPose.pose.position.y * 1000.0 << ", "
                      << moveitPose.pose.position.z * 1000.0 << "]\n";
            printPose("Moving relative to fixed", capture.movingRelativeToFixed);

            appendCsvRow(csv, captureId, jointRadians, moveitPose, capture);
        }

        tracker.shutdown();
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    rclcpp::shutdown();
    spinThread.join();
    return exitCode;
}
