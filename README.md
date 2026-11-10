# orb-slam3-rs (WORK IN PROGRESS)

Rust reimplementation of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). The original C++ ORB-SLAM3 is currently unmaintained and hard to build on newer systems while still being the most used reference for realtime SLAM. This project aims to provide an easier way to run and integrate ORB-SLAM3 into your projects. It follows C++ codebase relatively closely while trying to be as Rust idiomatic as possible and does aim to introduce some improvements: fixes for various segfaults the C++ codebase is sensitive to, multithreading improvements, bugfixes based on pull requests and small performance improvements.

For the optimizer the required slice of [g2o](https://github.com/RainerKuemmerle/g2o) functionality was ported to be contained within this implementation.

AI transparency: Significant parts of this port, especially the more complex mathematical code such as the optimizer and some unit tests for validating parity with the original C++ codebase, were written with the help of LLM tools. My understanding of the underlying math is limited, and correctness has only been validated empirically against the C++ implementation.

**Original authors:** Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, [José M. M. Montiel](http://webdiis.unizar.es/~josemari/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/).

### Related Publications:

[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, _IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021_. **[PDF](https://arxiv.org/abs/2007.11898)**.

[IMU-Initialization] Carlos Campos, J. M. M. Montiel and Juan D. Tardós, **Inertial-Only Optimization for Visual-Inertial Initialization**, _ICRA 2020_. **[PDF](https://arxiv.org/pdf/2003.05766.pdf)**

[ORBSLAM-Atlas] Richard Elvira, J. M. M. Montiel and Juan D. Tardós, **ORBSLAM-Atlas: a robust and accurate multi-map system**, _IROS 2019_. **[PDF](https://arxiv.org/pdf/1908.11585.pdf)**.

[ORBSLAM-VI] Raúl Mur-Artal, and Juan D. Tardós, **Visual-inertial monocular SLAM with map reuse**, IEEE Robotics and Automation Letters, vol. 2 no. 2, pp. 796-803, 2017. **[PDF](https://arxiv.org/pdf/1610.05949.pdf)**.

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. _IEEE Transactions on Robotics,_ vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://arxiv.org/pdf/1610.06475.pdf)**.

[Monocular] Raúl Mur-Artal, José M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. _IEEE Transactions on Robotics,_ vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](https://arxiv.org/pdf/1502.00956.pdf)**.

[DBoW2 Place Recognition] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. _IEEE Transactions on Robotics,_ vol. 28, no. 5, pp. 1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**

### Alternatives

For a newer greenfield, more permissively licensed Rust visual SLAM solution have a look at [visloc-rs](https://github.com/rsasaki0109/visloc-rs) which shows great promise.

# 1. License

orb-slam3-rs is, like the C++ source, released under GPLv3 license.

For a closed-source version of ORB-SLAM3 for commercial purposes, please contact the original authors: orbslam (at) unizar (dot) es.

If you use ORB-SLAM3 in an academic work, please cite:

    @article{ORBSLAM3_TRO,
      title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial
               and Multi-Map {SLAM}},
      author={Campos, Carlos AND Elvira, Richard AND G\´omez, Juan J. AND Montiel,
              Jos\'e M. M. AND Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={37},
      number={6},
      pages={1874-1890},
      year={2021}
     }

# 2. Prerequisites

You need a recent Rust compiler (1.85+, edition 2024), OpenCV 4.5 or newer,
and Clang/`libclang` (the `opencv` crate's build script parses the OpenCV
headers). `build.rs` probes `pkg-config --modversion opencv4` and adapts to the
`AlgorithmHint` argument that OpenCV 4.9 added, so both 4.5-4.8 and 4.9+ work;
set `OPENCV_PKGCONFIG_NAME` if your development package is not named `opencv4`.
The desktop viewer supports macOS and Linux through `eframe` + `wgpu`.

# 3. Building ORB-SLAM3-RS library and examples

```bash
git clone https://github.com/jlu-lujing/orb-slam3-rust.git
cd orb-slam3-rust
cargo build --release
```

The C++ reference is **not** vendored here. If you want to compare against it,
clone it next to the crate (it is not needed to build or run this one):

```bash
git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git ../ORB_SLAM3
```

The vocabulary is shipped compressed and unpacked by `build.rs` on the first
build, so the first build is slower than later ones.

# 4. Running ORB-SLAM3-RS with your camera

Command-line EuRoC stereo-inertial runner:

```bash
cargo run --bin stereo_inertial_euroc -- \
  vocabulary/orbvoc.txt \
  config/euroc.yaml \
  recording/euroc/MH_01_easy \
  recording/euroc/MH_01_easy/times.txt
```

The runner writes `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`.

RGB-D / TUM-style sequence:

```bash
cargo run --bin tum_rgbd -- \
  vocabulary/orbvoc.txt \
  path/to/camera.yaml \
  path/to/sequence \
  --associations path/to/associations.txt
```

Desktop viewer:

```bash
cargo run --features gui --bin slam_viewer -- \
  vocabulary/orbvoc.txt \
  config/euroc.yaml \
  recording/euroc/MH_01_easy \
  recording/euroc/MH_01_easy/times.txt \
  --realtime
```

The viewer runs SLAM on a worker thread and displays the current image and ORB
features, map points, keyframes, covisibility graph, and current camera pose.
It supports localization mode, reset, orbit/zoom navigation, track following,
left/right fisheye image display, inertial and loop graph overlays, single-step
debugging, drag-and-drop sequence loading, and trajectory/Atlas export.

The original C++ implementation lives at
<https://github.com/UZ-SLAMLab/ORB_SLAM3> and is not vendored in this repository
(see above).

The ORB vocabulary in `vocabulary/` is the one distributed with ORB-SLAM3
(`DBoW2` format) and carries the same GPLv3 terms.

See [PARITY_AUDIT.md](PARITY_AUDIT.md) for the current method-level comparison,
verified areas, intentional architectural differences, and remaining risks.

### Status

Measured against a reference ORB-SLAM3 build on the same data, this port is at or
ahead on every RGB-D sequence tried so far (six of seven clearly, one inside the
reference's own run-to-run spread) and tracks the long sequence end to end where
the reference loses a fifth of it. `PARITY_AUDIT.md` lists the exact numbers, the
single intentional divergence (`SearchLocalPoints` search window for RGB-D) and
the paths that could not be validated here (stereo, inertial, and loop closing
beyond a synthetic revisit sequence).

Contributions are welcome; run `cargo fmt`, `clippy -D warnings` and the test
suite before opening a pull request.

### Testing

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

The integration tests under `tests/` compare the ported optimizers against g2o
fixtures. CI runs the same commands on Ubuntu with the system OpenCV.

Quantitative trajectory comparison is available with:

```bash
cargo run --bin trajectory_eval -- \
  CameraTrajectory.txt groundtruth.txt --max-diff 0.02 --offset 0.0
```

EuRoC ground-truth CSV ordering is supported with `--euroc`. The tool reports
timestamp association count, Umeyama scale, ATE RMSE/mean/median/max, and RPE.
