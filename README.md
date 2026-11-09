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

Currently only tested on Linux. You just need a recent Rust compiler and OpenCV 4.x.

# 3. Building ORB-SLAM3-RS library and examples

Clone this repository:
https://github.com/nondescriptpointer/orb-slam3-rs

Build:
cargo build

# 4. Running ORB-SLAM3-RS with your camera

TBD
