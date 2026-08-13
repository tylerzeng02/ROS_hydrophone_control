// move_between_points: given a sequence of real, NDI-measured target points
// (collected by cyton_ndi_capture's ndi_measure, e.g. by jogging the arm
// around in RViz and pressing Enter at each spot), commands the arm to
// visit each one in turn via a RELATIVE Cartesian move computed from its
// live current position, and measures the real error at each stop.
//
// Rationale (from the conversation that motivated this tool): a relative
// "move by this much" command and an absolute "go to this point" command
// are literally the same MoveIt request under the hood (setPoseTarget with
// current_pose + delta vs. setPoseTarget with an absolute pose) -- the
// difference in expected accuracy comes from where the delta being
// commanded is ANCHORED, not from the command mechanism. Anchoring the
// delta to the arm's own LIVE current position at each hop (as this tool
// does) means each hop only has to be as good as the arm's own short-term
// repeatability, not its full absolute calibration accuracy -- and it
// self-corrects any drift from a prior hop instead of compounding it,
// since every hop re-measures "where am I actually right now" before
// computing the next move.
//
// A real subtlety this tool has to handle: the target points are recorded
// in the NDI tracker's own coordinate frame (moving marker relative to
// fixed marker), but MoveIt needs a delta expressed in ITS frame
// (base_link) to command a move. These two frames are rotated relative to
// each other by a real, substantial amount (verified this session: two
// independently-fit base-frame rotations, from different datasets, agreed
// to within ~3 degrees of each other as actual rotation matrices, despite
// their raw Euler-angle numbers looking wildly different -- Euler angles
// are famously non-unique, so always compare rotations as matrices, not
// raw angle triples). Only the ROTATION part of that frame relationship is
// needed here (translation cancels out for a delta/direction vector), so
// this tool is far less sensitive to the base-frame TRANSLATION
// uncertainty that plagues absolute-position tests elsewhere in this
// project.
//
// NdiTracker and its dependencies are copied verbatim from
// cyton_accuracy_check/src/run_accuracy_check.cpp / move_x_test.cpp
// (themselves copied from cyton_ndi_capture/src/ndi_measure.cpp) --
// already hardware-validated NDI connect/BX-polling/averaging code, not
// reimplemented here.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "moveit_msgs/msg/move_it_error_codes.hpp"

#include "ndicapi.h"

namespace {

// ---------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------
constexpr const char* DEFAULT_INPUT_CSV = "ndi_target_points.csv";
constexpr const char* DEFAULT_OUTPUT_CSV = "move_between_points_results.csv";
constexpr const char* DEFAULT_NDI_DEVICE = "/dev/ttyUSB1";
constexpr const char* DEFAULT_MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";
constexpr const char* DEFAULT_FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";

constexpr const char* PLANNING_GROUP = "arm";
// Same values established (and verified via the real move_group log, see
// CLAUDE.md/this session's history) for move_x_test's Cartesian-IK moves:
// a single 5s attempt was NOT reliably enough for elbow_yaw's narrow
// locked window to be found by random-seeded IK, and setNumPlanningAttempts
// alone doesn't multiply the total search time the way it sounds like it
// should (MoveIt runs a handful of threads IN PARALLEL sharing the same
// time budget, not sequential fresh-budget retries).
constexpr double PLANNING_TIME_SECONDS = 30.0;
constexpr unsigned int NUM_PLANNING_ATTEMPTS = 15;

constexpr int SETTLE_MS = 750;

constexpr int NDI_REQUIRED_VALID_SAMPLES = 30;
constexpr int NDI_SAMPLE_INTERVAL_MS = 20;
constexpr int REQUIRED_VISIBLE_MARKERS = 4;
constexpr double MAX_NDI_ERROR = 0.50;

// ---------------------------------------------------------------------
// NDI-frame -> MoveIt-frame ROTATION (only the rotation, not the full
// base-frame transform -- see the file header comment for why that's
// sufficient here).
//
// REFIT 2026-08-10 (calibration/current/refit_moveit_ndi_rotation.py):
// the previous batch2-derived rotation below was found to be stale --
// calibration/current/check_fixed_marker_drift.py measured ~6.31deg of
// drift in the fixed marker's own orientation since the batch2 session,
// and this independently-derived Kabsch fit (12 fresh paired NDI/MoveIt
// poses, collected via the 2026-08-10 ndi_measure change that logs
// MoveGroupInterface::getCurrentPose() alongside each NDI capture) agrees
// closely: 6.21deg between the old and new matrices, via a completely
// different method (delta-vector Kabsch fit vs. raw quaternion comparison)
// -- strong cross-validation that the drift is real, not a measurement
// artifact. RMS delta-vector error on the 12-pose fit data dropped from
// 6.03mm (old matrix) to 3.44mm (this one).
// Old (batch2-derived, now stale) rotation, kept for reference:
//     {-0.0352, 0.8862, 0.4619},
//     {0.6846, 0.3581, -0.6349},
//     {-0.7281, 0.2939, -0.6193},
//
// Convention (matches calibrate_kinematics.py's build_base_transform()):
// this matrix R maps a vector expressed in MoveIt/base_link coordinates to
// the same vector expressed in NDI/fixed-marker coordinates:
//     v_ndi = R * v_moveit
// To go the other way (we have a delta in NDI coordinates, need it in
// MoveIt coordinates to command a move), use the transpose (R is a proper
// rotation matrix, so R^-1 == R^T):
//     v_moveit = R^T * v_ndi
// Refit 2026-08-13 from moveit_ndi_accuracy_check_new13_replay_clean.csv (13
// valid paired poses, 1 getCurrentPose()-failed sentinel row auto-rejected).
// Only 0.46deg from the prior fit -- effectively confirms no meaningful
// marker drift, but deployed anyway per direct request.
constexpr double R_MOVEIT_TO_NDI[3][3] = {
    {0.0033, 0.8971, 0.4418},
    {0.6142, 0.3469, -0.7088},
    {-0.7891, 0.2737, -0.5499},
};

std::array<double, 3> rotateNdiDeltaToMoveIt(const std::array<double, 3>& deltaNdi) {
    // v_moveit = R^T * v_ndi -- i.e. dot deltaNdi with each COLUMN of R.
    std::array<double, 3> result{};
    for (int col = 0; col < 3; ++col) {
        double sum = 0.0;
        for (int row = 0; row < 3; ++row) {
            sum += R_MOVEIT_TO_NDI[row][col] * deltaNdi[row];
        }
        result[col] = sum;
    }
    return result;
}

// ---------------------------------------------------------------------
// From cyton_ndi_capture/src/ndi_measure.cpp (NdiTracker and dependencies)
// -- copied verbatim, same as move_x_test.cpp/run_accuracy_check.cpp.
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

// ---------------------------------------------------------------------
// Target-point loading -- reads whatever CSV cyton_ndi_capture's
// ndi_measure produced (or any CSV with the same moving_relative_fixed_tx_
// mm/_ty_mm/_tz_mm columns), one target point per row, in order.
// ---------------------------------------------------------------------

struct TargetPoint {
    double txMm = 0.0, tyMm = 0.0, tzMm = 0.0;
};

std::vector<TargetPoint> loadTargetPoints(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open input CSV: " + path);
    }

    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        throw std::runtime_error("Input CSV is empty: " + path);
    }

    std::vector<std::string> columns;
    {
        std::stringstream ss(headerLine);
        std::string field;
        while (std::getline(ss, field, ',')) {
            columns.push_back(field);
        }
    }

    int txCol = -1, tyCol = -1, tzCol = -1;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "moving_relative_fixed_tx_mm") txCol = static_cast<int>(i);
        if (columns[i] == "moving_relative_fixed_ty_mm") tyCol = static_cast<int>(i);
        if (columns[i] == "moving_relative_fixed_tz_mm") tzCol = static_cast<int>(i);
    }
    if (txCol < 0 || tyCol < 0 || tzCol < 0) {
        throw std::runtime_error(
            "Input CSV " + path + " is missing moving_relative_fixed_tx_mm/_ty_mm/_tz_mm columns "
            "-- expected the format written by cyton_ndi_capture's ndi_measure."
        );
    }

    std::vector<TargetPoint> points;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        if (static_cast<int>(fields.size()) <= std::max({txCol, tyCol, tzCol})) {
            std::cout << "WARNING: skipping malformed row: " << line << '\n';
            continue;
        }
        TargetPoint point;
        point.txMm = std::stod(fields[static_cast<std::size_t>(txCol)]);
        point.tyMm = std::stod(fields[static_cast<std::size_t>(tyCol)]);
        point.tzMm = std::stod(fields[static_cast<std::size_t>(tzCol)]);
        points.push_back(point);
    }

    return points;
}

double distanceMm(double dx, double dy, double dz) {
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [input_csv] [output_csv] [ndi_device] [moving_rom_path] [fixed_rom_path]\n"
              << "  input_csv must have moving_relative_fixed_tx_mm/_ty_mm/_tz_mm columns "
                 "(the format ndi_measure writes) -- one target point per row, visited in order.\n"
              << "  Defaults:\n"
              << "    input_csv:       " << DEFAULT_INPUT_CSV << '\n'
              << "    output_csv:      " << DEFAULT_OUTPUT_CSV << '\n'
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
    auto node = std::make_shared<rclcpp::Node>("move_between_points");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinThread([&executor]() { executor.spin(); });

    int exitCode = 0;
    try {
        std::cout << "Loading target points from " << inputCsv << "...\n";
        std::vector<TargetPoint> points = loadTargetPoints(inputCsv);
        std::cout << "Loaded " << points.size() << " target points.\n";
        if (points.empty()) {
            throw std::runtime_error("No target points loaded -- nothing to do.");
        }

        moveit::planning_interface::MoveGroupInterface moveGroup(node, PLANNING_GROUP);
        moveGroup.setPlanningTime(PLANNING_TIME_SECONDS);
        moveGroup.setNumPlanningAttempts(NUM_PLANNING_ATTEMPTS);

        std::cout << "Connecting to NDI tracker on " << ndiDevice << "...\n";
        NdiTracker tracker(ndiDevice, movingRom, fixedRom);
        tracker.initialize();

        std::ofstream csv(outputCsv, std::ios::out | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error("Could not open output CSV: " + outputCsv);
        }
        csv << "point_index,target_tx_mm,target_ty_mm,target_tz_mm,"
               "before_tx_mm,before_ty_mm,before_tz_mm,"
               "after_tx_mm,after_ty_mm,after_tz_mm,"
               "commanded_delta_ndi_mm_x,commanded_delta_ndi_mm_y,commanded_delta_ndi_mm_z,"
               "achieved_delta_ndi_mm_x,achieved_delta_ndi_mm_y,achieved_delta_ndi_mm_z,"
               "error_mm\n";

        std::vector<double> errors;
        errors.reserve(points.size());

        for (std::size_t i = 0; i < points.size(); ++i) {
            const TargetPoint& target = points[i];

            std::cout << "\n=== Point " << (i + 1) << "/" << points.size() << " ===\n"
                      << "  Target (NDI frame, mm): (" << target.txMm << ", " << target.tyMm << ", "
                      << target.tzMm << ")\n";

            std::cout << "  Measuring current position...\n";
            DualToolCapture before = tracker.collectBothTools();
            const NdiPoseSample& beforePose = before.movingRelativeToFixed;
            std::cout << "  Current (NDI frame, mm): (" << beforePose.txMm << ", " << beforePose.tyMm
                      << ", " << beforePose.tzMm << ")\n";

            const std::array<double, 3> deltaNdiMm = {{
                target.txMm - beforePose.txMm,
                target.tyMm - beforePose.tyMm,
                target.tzMm - beforePose.tzMm,
            }};
            const std::array<double, 3> deltaNdiM = {{
                deltaNdiMm[0] / 1000.0, deltaNdiMm[1] / 1000.0, deltaNdiMm[2] / 1000.0,
            }};
            const std::array<double, 3> deltaMoveItM = rotateNdiDeltaToMoveIt(deltaNdiM);

            std::cout << "  Commanded delta (NDI frame, mm): (" << deltaNdiMm[0] << ", "
                      << deltaNdiMm[1] << ", " << deltaNdiMm[2] << ")\n"
                      << "  Commanded delta (MoveIt frame, m): (" << deltaMoveItM[0] << ", "
                      << deltaMoveItM[1] << ", " << deltaMoveItM[2] << ")\n";

            geometry_msgs::msg::PoseStamped currentPose = moveGroup.getCurrentPose();
            geometry_msgs::msg::PoseStamped targetPose = currentPose;
            targetPose.pose.position.x += deltaMoveItM[0];
            targetPose.pose.position.y += deltaMoveItM[1];
            targetPose.pose.position.z += deltaMoveItM[2];

            moveGroup.setPoseTarget(targetPose);
            std::cout << "  Planning and executing...\n";
            auto result = moveGroup.move();

            if (result != moveit::core::MoveItErrorCode::SUCCESS) {
                std::cout << "  Move FAILED, error code " << result.val
                          << " -- skipping this point, continuing to the next.\n";
                continue;
            }

            std::cout << "  Move succeeded. Settling " << SETTLE_MS << "ms before capture...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(SETTLE_MS));

            DualToolCapture after = tracker.collectBothTools();
            const NdiPoseSample& afterPose = after.movingRelativeToFixed;
            std::cout << "  After (NDI frame, mm): (" << afterPose.txMm << ", " << afterPose.tyMm
                      << ", " << afterPose.tzMm << ")\n";

            const std::array<double, 3> achievedDeltaMm = {{
                afterPose.txMm - beforePose.txMm,
                afterPose.tyMm - beforePose.tyMm,
                afterPose.tzMm - beforePose.tzMm,
            }};
            const double errorMm = distanceMm(
                afterPose.txMm - target.txMm, afterPose.tyMm - target.tyMm, afterPose.tzMm - target.tzMm
            );
            errors.push_back(errorMm);

            std::cout << "  Error vs. target: " << errorMm << " mm\n";

            csv << i << ',' << target.txMm << ',' << target.tyMm << ',' << target.tzMm << ','
                << beforePose.txMm << ',' << beforePose.tyMm << ',' << beforePose.tzMm << ','
                << afterPose.txMm << ',' << afterPose.tyMm << ',' << afterPose.tzMm << ','
                << deltaNdiMm[0] << ',' << deltaNdiMm[1] << ',' << deltaNdiMm[2] << ','
                << achievedDeltaMm[0] << ',' << achievedDeltaMm[1] << ',' << achievedDeltaMm[2] << ','
                << errorMm << '\n';
            csv.flush();
        }

        if (!errors.empty()) {
            const double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
            const double mean = sum / static_cast<double>(errors.size());
            const double maxError = *std::max_element(errors.begin(), errors.end());
            const double sumSq =
                std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
            const double rms = std::sqrt(sumSq / static_cast<double>(errors.size()));

            std::cout << "\n=== Summary ===\n"
                      << errors.size() << " of " << points.size() << " points reached and measured.\n"
                      << "Mean error: " << mean << " mm\n"
                      << "RMS error:  " << rms << " mm\n"
                      << "Max error:  " << maxError << " mm\n"
                      << "Saved data to " << outputCsv << '\n';
        } else {
            std::cout << "\nNo points were successfully reached and measured.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        exitCode = 1;
    }

    executor.cancel();
    spinThread.join();
    rclcpp::shutdown();
    return exitCode;
}
