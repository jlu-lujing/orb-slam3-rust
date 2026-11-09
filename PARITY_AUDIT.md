# ORB-SLAM3 parity audit

Audit date: 2026-11-09

Reference: `3rdparty/ORB_SLAM3` at `4452a3c4ab75`

This document distinguishes code completeness from accuracy parity. The Rust
port is independently implemented and does not yet have a real-dataset
trajectory-accuracy comparison against the C++ build.

## Audit method

The audit compared:

- public classes and methods in `Atlas`, `Map`, `KeyFrame`, `MapPoint`,
  `Tracking`, `LocalMapping`, `LoopClosing`, `Optimizer`, and `System`;
- image preprocessing and trajectory-export branches;
- optimizer edge construction, fixed-vertex rules, robust kernels, and
  bias/preintegration handling;
- monocular, rectified-stereo, RGB-D, fisheye-rig, and inertial paths;
- local and loop-closing map maintenance;
- GUI functionality from the Pangolin viewers against the `eframe` viewer.

Numerical optimizer behavior is checked against generated g2o fixtures under
`tests/`. Other checks use focused Rust tests and an end-to-end synthetic EuRoC
smoke sequence.

## Implemented and locally verified

| Area | Status |
| --- | --- |
| ORB extraction and descriptor matching | Ported, unit-tested |
| Pinhole and Kannala-Brandt camera models | Ported, projection/Jacobian tests |
| Two-view reconstruction | Ported, planar/general/outlier tests |
| g2o core and ORB-SLAM3 optimizer edge types | Ported, analytic Jacobian tests |
| Global/local/pose/Sim3/inertial BA | Ported, g2o fixture parity tests |
| `FullInertialBA` fixed-local and shared-bias init modes | Wired, focused tests |
| IMU noise scaling, preintegration boundary interpolation | Aligned, focused tests |
| Monocular/stereo/RGB-D image preprocessing | Resize/remap wired, tests |
| Right fisheye-rig observations in inertial pose optimization | Wired, tests |
| Relocalization pose initialization | Bearing-based PnP-RANSAC path, pose-recovery test |
| TUM/EuRoC/KITTI trajectory semantics | Sensor/body/frame normalization aligned, tests |
| Active-map reset | Thread synchronization, DB cleanup, map detachment, tests |
| Bad-keyframe removal from BoW database | Implemented and tested |
| Keyframe BoW recomputation | Implemented and tested |
| Local mapping queue, BA, triangulation, fusion, culling, IMU init | Implemented |
| Loop closing detection, three-frame confirmation, Sim3 correction | Implemented |
| Loop search-and-fuse and global BA after correction | Implemented |
| Atlas/map merge with optional inertial merge BA | Implemented |
| Desktop GUI | `eframe` image/features, 3D map, graph, pause/step/reset/save |

## Intentional architecture differences

- Native C++ pointers and mutexes are replaced by `Arc`, `Weak`, `RwLock`,
  atomics, and explicit ownership boundaries.
- ORB-SLAM3 `g2o` is represented by an in-tree sparse optimizer with the edge
  and vertex types used by ORB-SLAM3.
- Pangolin/OpenGL viewers are replaced by a portable `eframe`/`wgpu` viewer.
- Atlas persistence uses `postcard` and is not binary-compatible with the C++
  Boost serialization format.
- Rust error values and typed APIs replace C++ return-code/`nullptr` patterns.

## Remaining differences and risks

These are the material areas where behavior is not yet an exact algorithm-level
match:

1. **Relocalization PnP implementation.** ORB-SLAM3 uses MLPnP directly on
   bearing vectors with covariance support. This port currently converts
   bearings to normalized image coordinates and runs OpenCV EPnP-RANSAC. This
   covers pinhole and fisheye front-facing observations, but is not a line-level
   MLPnP port and does not yet model per-bearing covariance.
2. **Loop candidate continuity.** A loop now requires three distinct,
   consecutive current keyframes to select the same candidate. The full
   ORB-SLAM3 common-region search across the previous loop group, including
   candidate-side continuity and retry bookkeeping, is still simplified.
3. **Global BA scheduling.** The port runs global BA synchronously after an
   essential-graph correction while LocalMapping is stopped. ORB-SLAM3 runs a
   dedicated global-BA thread and stages `mTcwGBA`/`mPosGBA` before committing.
4. **Map merge completeness.** Merge now moves contents, adds the loop edge,
   optionally runs inertial merge BA, and rebuilds connections. It does not yet
   reproduce every corrected-pose, duplicate-fusion, map-origin-id, and
   synchronization step of `MergeLocal`.
5. **Accuracy parity.** No real EuRoC/TUM/KITTI trajectory was available in this
   workspace. The synthetic EuRoC smoke test validates execution and exports,
   not ATE/RPE parity.
6. **C++ API/debug surface.** A few diagnostic methods such as timing injection
   and legacy Atlas overloads are intentionally absent or represented by Rust
   APIs.

## Verification commands

```bash
DYLD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib:/opt/homebrew/opt/llvm/lib:/opt/homebrew/Cellar/x265/4.1/lib \
LIBCLANG_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib \
cargo test --all-targets --features gui

DYLD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib:/opt/homebrew/opt/llvm/lib:/opt/homebrew/Cellar/x265/4.1/lib \
LIBCLANG_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib \
cargo clippy --all-targets --features gui -- -D warnings
```

The source-size counts are 29,902 C++ implementation lines versus 33,085 Rust
lines including inline tests and documentation. These counts are useful for
scope estimation only and are not a completion or equivalence metric.
