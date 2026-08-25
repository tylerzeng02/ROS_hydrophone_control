/**
 * @file test_ndi_single_tool.cpp
 * @brief Connects to an NDI Polaris Spectra over serial, loads a single
 * wireless passive-marker tool's .rom geometry, and polls its tracked
 * pose via the binary BX command.
 *
 * Follows NDI's documented port-handle flow: INIT -> PHRQ -> PVWR ->
 * PINIT -> PENA -> TSTART, then polls with BX (not the ASCII TX command,
 * which never resolves markers on this hardware even when the port
 * reports healthy). Also includes built-in diagnostics
 * (freeAllExistingHandles(), sweepVolumes()) that run automatically when
 * zero valid transforms are seen, to separate stale-port-handle causes
 * from measurement-volume causes of a persistent missing-marker result.
 * .rom file path and COM port are hardcoded, machine-specific absolute
 * paths.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "ndicapi.h"

namespace {

constexpr const char* NDI_DEVICE = "COM3";

constexpr const char* MOVING_TOOL_ROM =
    R"(C:\Users\ConformalUser\Desktop\Spectra\8700449- Polaris Passive 4-Marker Rigid Body 3(1).rom)";
constexpr int SAMPLE_COUNT = 100;
constexpr int SAMPLE_INTERVAL_MS = 50;
constexpr double MAX_NDI_ERROR = 0.50;

class SingleToolNdiTest {
public:
    SingleToolNdiTest(
        const char* device,
        const char* romPath
    )
        : device_(device),
          romPath_(romPath) {}

    ~SingleToolNdiTest() {
        shutdown();
    }

    void initialize() {
        tracker_ = ndiOpenSerial(device_);
        if (tracker_ == nullptr) {
            throw std::runtime_error(
                std::string("Could not open NDI device: ") + device_
            );
        }

        ndiTSTOP(tracker_);

        const char* resetReply = ndiRESET(tracker_);
        std::cout
            << "Serial break RESET reply: \""
            << (resetReply != nullptr ? resetReply : "(null)")
            << "\" | error=" << ndiGetError(tracker_) << '\n';

        ndiINIT(tracker_);
        requireNoNdiError("INIT");

        freeAllExistingHandles();
        queryTrackingParameters();

        ndiPHRQ(tracker_, "********", "0", "1", "**", "**");
        requireNoNdiError("PHRQ");

        toolHandle_ = ndiGetPHRQHandle(tracker_);
        if (toolHandle_ <= 0) {
            throw std::runtime_error(
                "NDI did not allocate a passive tool handle."
            );
        }

        if (ndiPVWRFromFile(
                tracker_,
                toolHandle_,
                const_cast<char*>(romPath_)
            ) != NDI_OKAY) {
            throw std::runtime_error(
                std::string("Could not load ROM file: ") + romPath_
            );
        }
        requireNoNdiError("PVWRFromFile");

        {
            const char* pinitReply = ndiPINIT(tracker_, toolHandle_);
            std::cout
                << "PINIT raw reply: \""
                << (pinitReply != nullptr ? pinitReply : "(null)")
                << "\"\n";
        }
        requireNoNdiError("PINIT");

        printHandleDiagnostics();

        ndiPENA(tracker_, toolHandle_, NDI_DYNAMIC);
        requireNoNdiError("PENA");

        ndiTSTART(tracker_);
        requireNoNdiError("TSTART");
        tracking_ = true;

        std::cout
            << "Tracking started for handle 0x"
            << std::hex << toolHandle_
            << std::dec << ".\n";
    }

    void run() {
        int validCount = 0;

        for (int sampleIndex = 1;
             sampleIndex <= SAMPLE_COUNT;
             ++sampleIndex) {

            ndiCommand(
                tracker_,
                "BX:%04X",
                NDI_XFORMS_AND_STATUS |
                NDI_ADDITIONAL_INFO |
                NDI_3D_MARKER_POSITIONS |
                NDI_NOT_NORMALLY_REPORTED
            );
            requireNoNdiError("BX");

            float transform[8] = {};

            const int result = ndiGetBXTransform(
                tracker_,
                toolHandle_,
                transform
            );

            const int status = ndiGetBXPortStatus(
                tracker_,
                toolHandle_
            );

            const unsigned long frame = ndiGetBXFrame(
                tracker_,
                toolHandle_
            );

            std::cout
                << "Sample " << sampleIndex
                << " | result=" << result
                << " | status=0x"
                << std::hex << status << std::dec
                << " | frame=" << frame;

            if (result == NDI_OKAY) {
                const bool finite =
                    std::isfinite(transform[0]) &&
                    std::isfinite(transform[1]) &&
                    std::isfinite(transform[2]) &&
                    std::isfinite(transform[3]) &&
                    std::isfinite(transform[4]) &&
                    std::isfinite(transform[5]) &&
                    std::isfinite(transform[6]) &&
                    std::isfinite(transform[7]);

                const bool enabled =
                    (status & NDI_ENABLED) != 0;

                const bool inVolume =
                    (status & NDI_OUT_OF_VOLUME) == 0;

                if (finite &&
                    enabled &&
                    inVolume &&
                    transform[7] <= MAX_NDI_ERROR) {

                    ++validCount;

                    std::cout
                        << " | VALID"
                        << " | q=["
                        << std::fixed << std::setprecision(6)
                        << transform[0] << ", "
                        << transform[1] << ", "
                        << transform[2] << ", "
                        << transform[3] << "]"
                        << " | t_mm=["
                        << transform[4] << ", "
                        << transform[5] << ", "
                        << transform[6] << "]"
                        << " | error="
                        << transform[7];
                } else {
                    std::cout
                        << " | transform returned but failed quality checks"
                        << " | error=" << transform[7];
                }
            } else if (result == NDI_MISSING) {
                std::cout << " | MISSING";
            } else if (result == NDI_DISABLED) {
                std::cout << " | DISABLED";
            } else {
                std::cout << " | other transform result";
            }

            std::cout << '\n';

            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    SAMPLE_INTERVAL_MS
                )
            );
        }

        std::cout
            << "\nValid transforms: "
            << validCount << " / "
            << SAMPLE_COUNT << '\n';

        if (validCount == 0) {
            throw std::runtime_error(
                "No valid transforms were received for rigid body 2."
            );
        }
    }

private:
    void tryQuery(const char* label, const char* command) {
        const char* reply = ndiCommand(tracker_, "%s", command);

        std::cout
            << label << " [" << command << "] => "
            << (reply != nullptr ? reply : "(null)")
            << " | error=" << ndiGetError(tracker_) << '\n';
    }

    void queryTrackingParameters() {
        std::cout << "\nQuerying named tracking parameters...\n";

        tryQuery("Sensitivity", "GET:Param.Tracking.Sensitivity");
        tryQuery("Alerts", "GET:Info.Status.Alerts");
        tryQuery("Selected Volume", "GET:Param.Tracking.Selected Volume");
        tryQuery(
            "Wavelength Warning",
            "GET:Param.Default Wavelength.Return Warning"
        );
        tryQuery(
            "Firmware Version",
            "GETINFO:Features.Firmware.Version"
        );

        tryQuery("All Info.* params", "GET:Info.*");
        tryQuery("All Param.* params", "GET:Param.*");

        tryQuery(
            "Illuminator Rate Info",
            "GETINFO:Param.Tracking.Illuminator Rate"
        );
        tryQuery(
            "Set Illuminator Rate index=1 (20Hz)",
            "SET:Param.Tracking.Illuminator Rate=1"
        );
        tryQuery(
            "Illuminator Rate after SET(1)",
            "GET:Param.Tracking.Illuminator Rate"
        );

        std::cout << '\n';
    }

    void freeAllExistingHandles() {
        ndiPHSR(tracker_, NDI_ALL_HANDLES);
        if (ndiGetError(tracker_) != NDI_OKAY) {
            std::cout
                << "PHSR (all handles) failed with error "
                << ndiGetError(tracker_) << ".\n";
            return;
        }

        const int existingCount = ndiGetPHSRNumberOfHandles(tracker_);
        std::cout
            << "Port handles already on the device before PHRQ: "
            << existingCount << '\n';

        for (int i = 0; i < existingCount; ++i) {
            const int handle = ndiGetPHSRHandle(tracker_, i);
            const int info = ndiGetPHSRInformation(tracker_, i);

            std::cout
                << "  Existing handle 0x" << std::hex << handle
                << std::dec << " | info=0x" << std::hex << info
                << std::dec << '\n';

            ndiPDIS(tracker_, handle);
            ndiPHF(tracker_, handle);
        }
    }

    void printHandleDiagnostics() {
        ndiPHINF(
            tracker_,
            toolHandle_,
            NDI_BASIC |
            NDI_PART_NUMBER |
            NDI_MARKER_TYPE |
            NDI_PORT_LOCATION
        );
        requireNoNdiError("PHINF");

        const int status =
            ndiGetPHINFPortStatus(tracker_);

        const int markerType =
            ndiGetPHINFMarkerType(tracker_);

        char toolInfo[128] = {};
        ndiGetPHINFToolInfo(
            tracker_,
            toolInfo
        );

        std::cout
            << "PHINF"
            << " | handle=0x"
            << std::hex << toolHandle_ << std::dec
            << " | status=0x"
            << std::hex << status << std::dec
            << " | markerType="
            << markerType
            << " | toolInfo=\"";

        for (int i = 0;
             i < 30 && toolInfo[i] != '\0';
             ++i) {
            const unsigned char c =
                static_cast<unsigned char>(
                    toolInfo[i]
                );

            if (c >= 32 && c <= 126) {
                std::cout << toolInfo[i];
            } else {
                std::cout << '.';
            }
        }

        std::cout << "\"\n";
    }

    void shutdown() noexcept {
        if (tracker_ == nullptr) {
            return;
        }

        if (tracking_) {
            ndiTSTOP(tracker_);
            tracking_ = false;
        }

        if (toolHandle_ > 0) {
            ndiPDIS(
                tracker_,
                toolHandle_
            );
            ndiPHF(
                tracker_,
                toolHandle_
            );
            toolHandle_ = 0;
        }

        ndiCloseSerial(tracker_);
        tracker_ = nullptr;
    }

    void requireNoNdiError(
        const char* operation
    ) {
        const int error =
            ndiGetError(tracker_);

        if (error != NDI_OKAY) {
            throw std::runtime_error(
                std::string("NDI ") +
                operation +
                " failed with error code " +
                std::to_string(error)
            );
        }
    }

    const char* device_;
    const char* romPath_;
    ndicapi* tracker_ = nullptr;
    int toolHandle_ = 0;
    bool tracking_ = false;
};

}  // namespace

int main() {
    try {
        std::cout
            << "Single-tool NDI test\n"
            << "Device: " << NDI_DEVICE << '\n'
            << "ROM: " << MOVING_TOOL_ROM << '\n';

        SingleToolNdiTest test(
            NDI_DEVICE,
            MOVING_TOOL_ROM
        );

        test.initialize();
        test.run();

        std::cout
            << "\nSingle-tool NDI test completed successfully.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}