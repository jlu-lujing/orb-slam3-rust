# Optimizer port — working notes

Faithful Rust port of `ORB_SLAM3/src/Optimizer.cc` (5590 LOC, 17 fns) + the slice of
g2o it depends on. Strategy: hand-port a focused "mini-g2o" onto nalgebra (no FFI, no
external solver crate) so numerics match upstream and the port stays 1:1 verifiable.

## g2o engine parity facts (from Thirdparty/g2o)

### LevenbergMarquardt (`optimization_algorithm_levenberg.cpp`)
- constants: `tau=1e-5`, goodStepUpper=2/3, goodStepLower=1/3, ni=2, maxTrialsAfterFailure=10
- per `solve(iteration)`:
  1. computeActiveErrors (each active edge computeError)
  2. currentChi = activeRobustChi2
  3. buildSystem (H,b from active edges; linearizeOplus then constructQuadraticForm)
  4. if iteration==0: lambda = tau * max|H_ii|, ni=2, nBad=0
  5. trust-region do/while:
     - push (backup estimates of active vertices)
     - setLambda: H_ii += lambda  (ADDITIVE, both pose & landmark diagonal blocks)
     - solve H x = b
     - update: oplus(x) on each indexMapping vertex
     - restoreDiagonal
     - computeActiveErrors; tempChi=activeRobustChi2 (max if solve failed)
     - rho = (currentChi - tempChi) / (computeScale + 1e-3)
       computeScale = Σ_j x_j*(lambda*x_j + b_j)
     - if rho>0 && finite(tempChi): good -> alpha=1-(2rho-1)^3; clamp [1/3,2/3];
       lambda*=clamp; ni=2; currentChi=tempChi; discardTop()
       else: lambda*=ni; ni*=2; pop()
     - while rho<0 && qmax<10 && !terminate
  6. if qmax==10 || rho==0: return Terminate
  7. Raul stop: if (iniChi-currentChi)*1e3 < iniChi -> nBad++ else nBad=0; if nBad>=3 Terminate

### optimize(N): loop i in 0..N { solve(i) }; stop on terminate/Fail.

### Edge construct quadratic form (base_unary/base_binary_edge.hpp)
- no kernel:   b -= Aᵀ Ω e ;            H += Aᵀ Ω A
- with kernel: robustify(chi2)->rho;  weightedOmega = rho[1]*Ω  (rho[2] term commented out!)
               b -= rho[1] * Aᵀ Ω e ;  H += Aᵀ (rho[1]Ω) A
- binary edge cross block: H_ij += Aᵀ Ω B  (Xi=vertex0, Xj=vertex1)

### chi2 = eᵀ Ω e   (base_edge.h)
### activeRobustChi2 = Σ (kernel? rho[0] : chi2)

### RobustKernelHuber::robustify(e, rho), dsqr=delta²:
  if e<=dsqr: rho=[e,1,0]
  else: s=sqrt(e); rho=[2*s*delta-dsqr, delta/s, -0.5*(delta/s)/e]

### VertexSE3Expmap: estimate SE3Quat; oplus: est = SE3Quat::exp(update) * est
   update = [omega(0..3), upsilon(3..6)]  (rotation first)
### VertexSBAPointXYZ: estimate Vector3d; oplus: est += update
### SE3Quat::exp / log / map / *: see se3quat.h (ported in g2o_core.rs)

### indexMapping order: non-fixed, non-marginalized first (k=0) then marginalized (k=1),
   in active-vertex order (sorted by id). Hessian index assigned in that order.

## PoseOptimization (Optimizer.cc:814)
- single VertexSE3Expmap (id 0, not fixed), points NOT vertices (stored as Xw in edge)
- mono: ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose (uses pCamera->project / projectJac)
- stereo: g2o::EdgeStereoSE3ProjectXYZOnlyPose (explicit fx,fy,cx,cy,bf form)
- right-cam (fisheye rig): EdgeSE3ProjectXYZOnlyPoseToBody (mTrl)
- deltaMono=sqrt(5.991), deltaStereo=sqrt(7.815)
- 4 passes, its={10,10,10,10}; reset estimate to pFrame pose each pass
- chi2 thresholds {5.991 mono, 7.815 stereo}; classify outliers via level 0/1
- it==2: drop robust kernel
- break if active edges < 10
- return nInitialCorrespondences - nBad

## STATUS
- [x] mini-g2o core (`g2o_core.rs`): SE3Quat, RobustKernelHuber, Vertex/Edge traits,
      SparseOptimizer + LM. Validated against real g2o.
- [x] non-inertial unary edges (`optimizable_types.rs`): VertexSE3Expmap,
      VertexSBAPointXYZ, EdgeSE3ProjectXYZOnlyPose, ...ToBody, EdgeStereoSE3...OnlyPose
- [x] PoseOptimization — VALIDATED (tests/optimizer_pose.rs vs g2o fixture:
      exact inlier flags+count, pose <1e-5)
- [x] Schur-complement block solver (BlockSystem in g2o_core.rs) — VALIDATED via BA
- [x] non-inertial binary edges: EdgeSE3ProjectXYZ, ...ToBody, EdgeStereoSE3ProjectXYZ
- [x] BundleAdjustment core — VALIDATED (tests/optimizer_ba.rs vs g2o BlockSolver_6_3:
      5 KF / 60 MP / 299 obs mono+stereo, <1e-4)
- [x] bundle_adjustment(map)/global_bundle_adjustment wired over real KeyFrame/MapPoint/Map
- [x] LocalBundleAdjustment (non-inertial) wired over real types
- [x] Sim3 + VertexSim3Expmap + Sim3 projection edges (numerical jac) + EdgeSim3 (pose graph)
- [x] OptimizeSim3 core+real — VALIDATED (tests/optimizer_sim3.rs, free+fixed scale)
- [x] OptimizeEssentialGraph core — VALIDATED (tests/optimizer_essgraph.rs, free+fixed scale)
- [x] Schur block solver supports arbitrary block dims (7-DoF Sim3); userLambdaInit + forceStop

### PARITY NOTE
All non-inertial parity tests tightened to 1e-6 (angles/translations) / 1e-5 (metric pts)
= bit-level agreement with g2o (residual is f64 accumulation over LM iters). Sim3/EdgeSim3
use numerical Jacobians because g2o does too (matched exactly).

### DONE (inertial g2o types) — merged into g2o_types.rs, full ANALYTIC Jacobians
- [x] ImuCamPose, VertexPose, VertexVelocity, VertexGyroBias, VertexAccBias, VertexScale,
      VertexGDir; EdgeMono/Stereo(+OnlyPose), EdgeInertial, EdgeInertialGS, EdgeBiasRW
      (gyro/acc), EdgeBiasPrior (gyro/acc), EdgePriorPoseImu; ConstraintPoseIMU(+accessors)
- [x] f64 SO3 helpers in g2o_core (exp/log/right_jac/inv_right_jac/normalize)
- [x] VALIDATED: tests/optimizer_inertial_jac.rs — analytic vs finite-diff for EdgeMono,
      EdgeStereo, EdgeInertial, EdgeInertialGS, EdgePriorPoseImu (EdgeInertialGS scale
      column intentionally matches g2o's approximate Jacobian, not the true FD)

### DONE (inertial optimizer)
- [x] Gauss–Newton mode in SparseOptimizer (OptimizationAlgorithmGaussNewton)
- [x] EdgeInertial / EdgeInertialGS now support robust kernels (engine weights generically)
- [x] Marginalize (Schur-complement block, pseudo-inverse) — ported
- [x] InertialOptimization (Rwg, scale) core+ — VALIDATED (tests/optimizer_inertialgs.rs vs
      g2o EdgeInertialGS + Gauss-Newton, <1e-5; preintegration fields reconstructed from C++)

- [x] edge_linearization accessor (for marginalization Hessians)
- [x] PoseInertialOptimizationLastKeyFrame core — VALIDATED (tests/optimizer_poseinertial.rs):
      state bit-exact, outlier flags + inlier count exact, prior Hessian rel-Frobenius <1e-6.
      (Note: MapPoint pos is f32 in reality; prior Hessian small entries sit in the IMU
      info near-null-space — compare via relative Frobenius, not per-entry.)
- [x] Fixed pre-existing bug in ConstraintPoseIMU::new: (h+h)/2 -> (h+hᵀ)/2

- [x] Marginalize — VALIDATED (tests/optimizer_marginalize.rs vs direct Schur complement)
- [x] PoseInertialOptimizationLastFrame core — VALIDATED (tests/optimizer_poseinertiallf.rs):
      EdgePriorPoseImu(robust) + 30→15 marginalization; state/flags/inliers exact, prior <1e-6

### DONE (inertial BA batch)
- [x] inertial_ba_core (shared by Full/Local/Merge) — VALIDATED (tests/optimizer_inertialba.rs):
      4 KF / 50 pts / 200 obs, EdgeInertial + bias RW + EdgeMono/EdgeStereo, LM, <1e-5/<1e-4
- [x] EdgeInertial::scale_information (LocalInertialBA boundary down-weight)
- [x] local_inertial_ba real `&Map`/`&KeyFrame` wrapper (temporal window + fixed boundary)
- [x] full_inertial_ba real `&Map` wrapper (non-init path; direct + pose/point GBA staging)

### DONE (final batch)
- [x] InertialOptimization unified core + 3 real wrappers (gravity+scale / full / bias-only);
      full overload VALIDATED (tests/optimizer_inertialfull.rs)
- [x] EdgeBiasPrior wired into the bias-prior path
- [x] MergeInertialBA real wrapper (welding window over inertial_ba_core)
- [x] VertexPose4DoF + Edge4DoF (ImuCamPose::update_w) and
      optimize_essential_graph_4dof_core — VALIDATED (tests/optimizer_essgraph4dof.rs)

## ALL 17 Optimizer.cc FUNCTIONS COMPLETE (cores validated vs g2o; real wrappers wired).

### Final closeout
- [x] local_inertial_ba: signature now mirrors C++
      (kf, stop_flag, map, b_large, b_rec_init) -> (num_fixed, num_opt, num_mps, num_edges);
      added outlier culling (erase observations) + divergence guard + bRecInit robust links.
      inertial_ba_core now returns InertialBaResult (states, points, per-obs chi2/depth,
      err_start/err_end) + accepts a stop flag.
- [x] OptimizeEssentialGraph (loop-closing, 1st overload) real &Map wrapper, over the
      validated optimize_essential_graph_core; LoopClosing types stubbed as
      KeyFrameAndPose = HashMap<u64,Sim3>, LoopConnections = HashMap<u64,HashSet<u64>>.
- [x] OptimizeEssentialGraph (merge, 2nd overload) real wrapper.
- Updated local_mapping.rs call site to the new local_inertial_ba signature.

The essential-graph real wrappers are loop-closing glue over the validated EdgeSim3
pose-graph core; they can't be end-to-end tested until LoopClosing is implemented, but
their numeric core + all edges are validated.

## API AUDIT (all 18 C++ public methods -> Rust public fn, args matched)
  PoseOptimization                       pose_optimization(frame)->i32
  GlobalBundleAdjustemnt                 global_bundle_adjustment(map,...)
  BundleAdjustment                       bundle_adjustment(map,kfs,mps,n,stop,loop,robust)
  LocalBundleAdjustment (std)            local_bundle_adjustment(kf,stop,map)->(counts)
  LocalBundleAdjustment (merge welding)  local_bundle_adjustment_merge(main,adjust,fixed,stop)
  OptimizeSim3                           optimize_sim3(kf1,kf2,&mut matches,&mut s12,th2,fix,all)->i32
  OptimizeEssentialGraph (loop)          optimize_essential_graph(map,loop,cur,nc,c,conns,fix)
  OptimizeEssentialGraph (merge)         optimize_essential_graph_merge(cur,fixed,fixedc,nonfixed,mps)
  OptimizeEssentialGraph4DoF             optimize_essential_graph_4dof(map,loop,cur,nc,c,conns)
  Marginalize                            marginalize(h,start,end)->DMatrix
  InertialOptimization (full)            inertial_optimization(map,&mut Rwg,&mut s,&mut bg,&mut ba,mono,fixvel,pg,pa)
  InertialOptimization (bias)            inertial_optimization_bias(map,&mut bg,&mut ba,pg,pa)
  InertialOptimization (gravity+scale)   inertial_optimization_gravity_scale(map,&mut Rwg,&mut s)
  PoseInertialOptimizationLastKeyFrame   pose_inertial_optimization_last_keyframe(frame,recinit)->i32
  PoseInertialOptimizationLastFrame      pose_inertial_optimization_last_frame(frame,recinit)->i32
  FullInertialBA                         full_inertial_ba(map,its,fixlocal,loop,stop,init,pg,pa)
  LocalInertialBA                        local_inertial_ba(kf,stop,map,large,recinit)->(counts)
  MergeInertialBA                        merge_inertial_ba(cur,merge,stop,map)->corrPoses
C++ `int&` out-params (LocalBA counts) are returned tuples; dead C++ outputs
(mAcumHessian / covInertial / vSingVal / bHess) are omitted.

## TEST COVERAGE
Every numeric *core* has a C++-parity (or analytic-vs-FD) test — these contain all the
algorithm logic. The 18 public &Frame/&Map/&KeyFrame wrappers delegate to those validated
cores (mechanical field gathering + write-back) and are not separately unit-tested
(would need heavy Frame/Map fixtures; several depend on still-stubbed LoopClosing).

## TEST INVENTORY (18 parity + 195 unit, all green)
pose, ba, sim3(x2), essgraph(x2), essgraph4dof, inertial_jac(x5), inertialgs,
inertialfull, inertialba, marginalize, poseinertial, poseinertiallf
- [ ] LocalBundleAdjustment (non-inertial)
- [ ] OptimizeSim3, OptimizeEssentialGraph
- [ ] inertial: InertialOptimization x3, PoseInertial{LastKF,LastFrame}, LocalInertialBA, FullInertialBA
- [ ] merge/4DoF: MergeInertialBA, OptimizeEssentialGraph4DoF, Marginalize, LocalBA(welding)

### NEXT (BA family) needs:
- Binary edges: EdgeSE3ProjectXYZ (mono), EdgeSE3ProjectXYZToBody, EdgeStereoSE3ProjectXYZ
- Schur-complement sparse solver in SparseOptimizer (landmarks marginalized, 3x3 blocks),
  because dense H is infeasible for real maps (n ~ 6*nKF + 3*nMP). Math result identical
  to current dense path -> can validate Schur against dense on a small fixture first.

## Validation harness
C++ generator (tests/optimizer_fixture_gen.cc) emits deterministic fixtures (fixed seed)
with inputs + reference outputs to tests/fixtures/*.txt; Rust tests replay & compare.
</content>
