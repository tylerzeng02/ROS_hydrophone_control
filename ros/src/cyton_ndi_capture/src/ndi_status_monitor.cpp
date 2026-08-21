// ndi_status_monitor: minimal marker-visibility check. Connects to the NDI
// tracker, loads both tool ROMs, then polls forever and prints each tool's
// status only when it changes. Purely a "can the tracker currently see
// my markers" diagnostic, with no capture, no CSV, no ROS dependency (it
// does not need joint states, so it is plain C++ against ndicapi only).
// Ctrl+C to quit.
//
// Deliberately a separate, smaller program from ndi_measure rather than a
// mode flag on it. This one never blocks waiting for visibility, since
// the whole point is watching the status become visible, and it never
// averages or logs a capture.

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "ndicapi.h"

namespace {

constexpr const char* DEFAULT_NDI_DEVICE = "/dev/ttyUSB1";
constexpr const char* DEFAULT_MOVING_TOOL_ROM =
    "/home/temp/Downloads/8700339- Polaris Passive 4-Marker Rigid Body 2(1).rom";
constexpr const char* DEFAULT_FIXED_TOOL_ROM =
    "/home/temp/Downloads/8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom";

constexpr int POLL_INTERVAL_MS = 100;
constexpr double MAX_NDI_ERROR = 0.50;  // same threshold as ndi_measure/ndi_capture_and_validate

enum class NdiToolStatus { Detected, Missing, OutOfVolume, Disabled, LowQuality };

const char* toolStatusLabel(NdiToolStatus status) {
    switch (status) {
        case NdiToolStatus::Detected:    return "DETECTED";
        case NdiToolStatus::Missing:     return "MISSING";
        case NdiToolStatus::OutOfVolume: return "OUT_OF_VOLUME";
        case NdiToolStatus::Disabled:    return "DISABLED";
        case NdiToolStatus::LowQuality:  return "LOW_QUALITY (tracked, but error over threshold)";
    }
    return "UNKNOWN";
}

void requireNoNdiError(ndicapi* tracker, const char* operation) {
    const int error = ndiGetError(tracker);
    if (error != NDI_OKAY) {
        throw std::runtime_error(
            std::string("NDI ") + operation + " failed with error code " + std::to_string(error)
        );
    }
}

int allocateAndInitializeTool(ndicapi* tracker, const char* romPath, const char* toolName) {
    ndiPHRQ(tracker, "********", "0", "1", "**", "**");
    requireNoNdiError(tracker, "PHRQ");

    const int handle = ndiGetPHRQHandle(tracker);
    if (handle <= 0) {
        throw std::runtime_error(std::string("NDI did not allocate the ") + toolName + " tool handle.");
    }

    if (ndiPVWRFromFile(tracker, handle, const_cast<char*>(romPath)) != NDI_OKAY) {
        throw std::runtime_error(std::string("Could not load ") + toolName + " ROM file: " + romPath);
    }
    requireNoNdiError(tracker, "PVWRFromFile");

    ndiPINIT(tracker, handle);
    requireNoNdiError(tracker, "PINIT");

    ndiPENA(tracker, handle, NDI_DYNAMIC);
    requireNoNdiError(tracker, "PENA");

    std::cout << "Loaded and enabled " << toolName << " tool ROM into handle 0x" << std::hex << handle
              << std::dec << ".\n";
    return handle;
}

NdiToolStatus classifyToolStatus(ndicapi* tracker, int toolHandle) {
    float transform[8] = {};
    const int result = ndiGetBXTransform(tracker, toolHandle, transform);

    if (result == NDI_DISABLED) return NdiToolStatus::Disabled;
    if (result == NDI_MISSING) return NdiToolStatus::Missing;

    const int status = ndiGetBXPortStatus(tracker, toolHandle);
    if ((status & NDI_OUT_OF_VOLUME) != 0) return NdiToolStatus::OutOfVolume;
    if (!std::isfinite(transform[7]) || transform[7] > MAX_NDI_ERROR) return NdiToolStatus::LowQuality;
    return NdiToolStatus::Detected;
}

void printIfChanged(
    const char* toolName, NdiToolStatus current, NdiToolStatus& lastPrinted, bool& everPrinted
) {
    if (everPrinted && current == lastPrinted) {
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const auto nowT = std::chrono::system_clock::to_time_t(now);
    char timeBuf[16];
    std::strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", std::localtime(&nowT));

    std::cout << "[" << timeBuf << "] " << toolName << ": " << toolStatusLabel(current) << '\n';
    lastPrinted = current;
    everPrinted = true;
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [ndi_device] [moving_rom_path] [fixed_rom_path]\n"
              << "  Defaults:\n"
              << "    ndi_device:      " << DEFAULT_NDI_DEVICE << '\n'
              << "    moving_rom_path: " << DEFAULT_MOVING_TOOL_ROM << '\n'
              << "    fixed_rom_path:  " << DEFAULT_FIXED_TOOL_ROM << '\n';
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

    // Force a flush after every '<<'. Without this, std::cout is fully
    // buffered (not line-buffered) whenever stdout is not a TTY (piped,
    // redirected to a file, or captured by a launcher/log tool), so output
    // silently sits in the buffer and never reaches the log until the
    // process exits normally. This program is meant to be watched live
    // (or killed with Ctrl+C, which never lets buffered output flush), so
    // that default is actively wrong here.
    std::cout.setf(std::ios::unitbuf);

    ndicapi* tracker = nullptr;
    int movingHandle = 0;
    int fixedHandle = 0;

    try {
        std::cout << "Connecting to NDI tracker on " << device << "...\n";
        tracker = ndiOpenSerial(device);
        if (tracker == nullptr) {
            throw std::runtime_error(std::string("Could not open NDI device: ") + device);
        }

        ndiTSTOP(tracker);
        ndiINIT(tracker);
        requireNoNdiError(tracker, "INIT");

        movingHandle = allocateAndInitializeTool(tracker, movingRom, "moving");
        fixedHandle = allocateAndInitializeTool(tracker, fixedRom, "fixed");

        ndiTSTART(tracker);
        requireNoNdiError(tracker, "TSTART");

        std::cout << "\nTracking started. Watching status (Ctrl+C to quit)...\n\n";

        NdiToolStatus lastMoving = NdiToolStatus::Detected;
        NdiToolStatus lastFixed = NdiToolStatus::Detected;
        bool movingPrinted = false;
        bool fixedPrinted = false;

        while (true) {
            ndiCommand(
                tracker, "BX:%04X",
                NDI_XFORMS_AND_STATUS | NDI_ADDITIONAL_INFO | NDI_3D_MARKER_POSITIONS |
                    NDI_NOT_NORMALLY_REPORTED
            );
            requireNoNdiError(tracker, "BX");

            printIfChanged("Moving tool", classifyToolStatus(tracker, movingHandle), lastMoving, movingPrinted);
            printIfChanged("Fixed tool ", classifyToolStatus(tracker, fixedHandle), lastFixed, fixedPrinted);

            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << '\n';
        if (tracker != nullptr) {
            ndiTSTOP(tracker);
            if (movingHandle > 0) { ndiPDIS(tracker, movingHandle); ndiPHF(tracker, movingHandle); }
            if (fixedHandle > 0) { ndiPDIS(tracker, fixedHandle); ndiPHF(tracker, fixedHandle); }
            ndiCloseSerial(tracker);
        }
        return 1;
    }
}
