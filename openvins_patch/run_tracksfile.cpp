/*
 * run_tracksfile.cpp -- headless OpenVINS driver fed by EXTERNAL feature tracks.
 *
 * Approach A, Phase A2. Instead of running OpenVINS' own KLT on images, this
 * reads pre-computed tracks (exported from our Python front-ends -- classical
 * KLT or learned SuperPoint+LightGlue) and pushes them into the MSCKF back-end
 * through VioManager::feed_measurement_simulation. The filter is otherwise
 * untouched, so running this twice (KLT tracks vs learned tracks) and comparing
 * trajectory ATE isolates exactly the front-end's contribution to VIO accuracy.
 *
 * Inputs (all produced by scripts/phaseA2_export_tracks.py):
 *   config.yaml   OpenVINS mono config (config/euroc_mono)
 *   imu.txt       t wx wy wz ax ay az                       (>= init time)
 *   tracks.txt    t feat_id u v      (raw distorted pixels, time-sorted)
 *   gt_init.txt   t / p(3) / R_GtoI(9 row-major) / v(3) / bg(3) / ba(3)
 *   out_est.txt   TUM trajectory written here: t px py pz qx qy qz qw
 *
 * Derived from OpenVINS ov_msckf/src/run_simulation.cpp (GPLv3). Only the data
 * source changes (files instead of the Simulator); the filter calls are the same.
 * This derived file is likewise GPLv3.
 */
#include <Eigen/Eigen>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

// One camera frame's feature observations: (feature id, raw pixel uv).
struct CamFrame {
  double t = -1;
  std::vector<std::pair<size_t, Eigen::VectorXf>> feats;
};

int main(int argc, char **argv) {
  if (argc < 6) {
    std::printf("usage: run_tracksfile <config.yaml> <imu.txt> <tracks.txt> "
                "<gt_init.txt> <out_est.txt>\n");
    return EXIT_FAILURE;
  }
  std::string config_path = argv[1];
  std::string imu_path = argv[2];
  std::string tracks_path = argv[3];
  std::string init_path = argv[4];
  std::string out_path = argv[5];

  // ---- params + manager (YamlParser reads the file directly, no ROS) ----
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  auto sys = std::make_shared<VioManager>(params);

  // ---- read GT init state; convert R_GtoI -> JPL quat with OpenVINS' own util ----
  std::ifstream fi(init_path);
  double t_init;
  Eigen::Vector3d p, v, bg, ba;
  Eigen::Matrix3d R_GtoI;
  fi >> t_init;
  fi >> p(0) >> p(1) >> p(2);
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) fi >> R_GtoI(r, c);
  fi >> v(0) >> v(1) >> v(2);
  fi >> bg(0) >> bg(1) >> bg(2);
  fi >> ba(0) >> ba(1) >> ba(2);
  fi.close();

  Eigen::Matrix<double, 4, 1> q_GtoI = ov_core::rot_2_quat(R_GtoI);
  Eigen::Matrix<double, 17, 1> imustate;   // [t, q_GtoI, p, v, bg, ba]
  imustate(0) = t_init;
  imustate.block(1, 0, 4, 1) = q_GtoI;
  imustate.block(5, 0, 3, 1) = p;
  imustate.block(8, 0, 3, 1) = v;
  imustate.block(11, 0, 3, 1) = bg;
  imustate.block(14, 0, 3, 1) = ba;
  sys->initialize_with_gt(imustate);

  // ---- load IMU ----
  std::vector<ov_core::ImuData> imu;
  {
    std::ifstream f(imu_path);
    double t, wx, wy, wz, ax, ay, az;
    while (f >> t >> wx >> wy >> wz >> ax >> ay >> az) {
      ov_core::ImuData m;
      m.timestamp = t;
      m.wm << wx, wy, wz;
      m.am << ax, ay, az;
      imu.push_back(m);
    }
  }

  // ---- load tracks, grouped by timestamp (the export is time-sorted) ----
  std::vector<CamFrame> cams;
  {
    std::ifstream f(tracks_path);
    double t, u, vv;
    long id;
    CamFrame cf;
    while (f >> t >> id >> u >> vv) {
      if (t != cf.t) {
        if (cf.t >= 0) cams.push_back(std::move(cf));
        cf = CamFrame();
        cf.t = t;
      }
      Eigen::VectorXf uv(2);
      uv << static_cast<float>(u), static_cast<float>(vv);
      cf.feats.emplace_back(static_cast<size_t>(id), uv);
    }
    if (cf.t >= 0) cams.push_back(std::move(cf));
  }
  std::printf("[run_tracksfile] %zu imu, %zu cam frames, init t=%.3f\n",
              imu.size(), cams.size(), t_init);

  // ---- merge-feed by time; record pose after each camera update ----
  std::ofstream out(out_path);
  out << std::fixed << std::setprecision(9);
  size_t ii = 0, saved = 0;
  for (auto &cf : cams) {
    while (ii < imu.size() && imu[ii].timestamp <= cf.t) {
      sys->feed_measurement_imu(imu[ii]);
      ii++;
    }
    std::vector<int> camids = {0};
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
    feats.push_back(cf.feats);
    sys->feed_measurement_simulation(cf.t, camids, feats);

    if (sys->initialized()) {
      auto state = sys->get_state();
      Eigen::Vector3d pos = state->_imu->pos();
      Eigen::Matrix3d R_ItoG = ov_core::quat_2_Rot(state->_imu->quat()).transpose();
      Eigen::Quaterniond qh(R_ItoG);   // Hamilton quaternion for TUM output
      out << state->_timestamp << " " << pos(0) << " " << pos(1) << " " << pos(2)
          << " " << qh.x() << " " << qh.y() << " " << qh.z() << " " << qh.w()
          << "\n";
      saved++;
    }
  }
  out.close();
  std::printf("[run_tracksfile] wrote %zu poses -> %s\n", saved, out_path.c_str());
  return EXIT_SUCCESS;
}
