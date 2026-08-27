# build/

This is the CMake build directory (`cmake -S . -B build -G "NMake Makefiles"`)
— compiled `.exe`/`.pdb`/`.ilk` files and CMake's own cache/generated files
live here, none of it tracked in git.

**No calibration data is checked into this directory anymore.** It used to
hold the real, tracked CSVs behind the deployed 48-param calibration model —
those now live in `calibration/data/` instead, specifically so this
directory can be what its name says: build output, not a home for real
project data. See `calibration/data/README.md`.

## A quirk worth knowing: hardware tools still *write* here by default

The native hardware tools (`ndi_capture_and_validate.exe`,
`record_hand_poses.exe`, etc.) are built into this directory
and, per this project's own build instructions, are meant to be run from
inside it (`cd build`, then run the `.exe` directly) — so any output file
they write with a bare relative filename (no directory prefix) lands here by
default, not in `calibration/data/` or `pid_tuning/data/`.

**If you rerun a data-collection tool and want the result kept long-term,
move the output CSV into `calibration/data/` (or `pid_tuning/data/` for
I-gain-related collections) afterward** — the same manual step this
project's data has always needed; nothing here does that relocation
automatically. `archive/` (gitignored, kept locally only) holds every
historical output that's accumulated here this way over the project's life.
