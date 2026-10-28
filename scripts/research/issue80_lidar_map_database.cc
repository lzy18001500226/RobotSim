// Build a static RAICOM LiDAR signature map without starting MC or stepping.
// The known scene is sampled offline at fixed map poses using mj_ray.  Runtime
// localization consumes only local sensor scans against this fixed database.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

struct Candidate {
  float x;
  float y;
  float yaw;
  std::vector<float> range;
};

static int RootBody(const mjModel* model, int body) {
  while (body > 0 && model->body_parentid[body] > 0) body = model->body_parentid[body];
  return body;
}

static mjtNum RaySkippingRobot(const mjModel* model, const mjData* data,
                               const mjtNum origin[3], const mjtNum direction[3],
                               int root, mjtNum maximum) {
  mjtNum shifted[3] = {origin[0], origin[1], origin[2]};
  mjtNum travelled = 0;
  for (int attempt = 0; attempt < 64 && travelled < maximum; ++attempt) {
    int geom = -1;
    const mjtNum hit = mj_ray(model, data, shifted, direction, nullptr, 1, -1, &geom);
    if (hit < 0 || geom < 0) return -1;
    if (RootBody(model, model->geom_bodyid[geom]) != root) return travelled + hit;
    const mjtNum advance = hit + 1e-3;
    travelled += advance;
    for (int axis = 0; axis < 3; ++axis) shifted[axis] += advance * direction[axis];
  }
  return -1;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: lidar_map_database <plugin_dir> <scene_xml> <database_bin> <result_json>\n";
    return 2;
  }
  const std::string plugin_dir(argv[1]);
  const std::string scene(argv[2]);
  const std::string output_path(argv[3]);
  const std::string result_path(argv[4]);
  mj_loadAllPluginLibraries(plugin_dir.c_str(), nullptr);
  char error[4096] = {};
  mjModel* model = mj_loadXML(scene.c_str(), nullptr, error, sizeof(error));
  if (!model) {
    std::ofstream result(result_path);
    result << "{\"status\":\"FAIL_MODEL_LOAD\",\"error\":\"" << error << "\"}\n";
    return 3;
  }
  mjData* data = mj_makeData(model);
  const int free_joint = mj_name2id(model, mjOBJ_JOINT, "floating_base_joint");
  const int camera = mj_name2id(model, mjOBJ_CAMERA, "RayCasterLidar");
  if (free_joint < 0 || camera < 0) {
    std::ofstream result(result_path);
    result << "{\"status\":\"FAIL_REQUIRED_NAME\"}\n";
    mj_deleteData(data); mj_deleteModel(model); return 4;
  }
  const int qpos = model->jnt_qposadr[free_joint];
  const int root = RootBody(model, model->cam_bodyid[camera]);
  constexpr int horizontal = 90;
  constexpr int vertical = 5;
  constexpr float horizontal_fov = 270.0f;
  constexpr float vertical_fov = 60.0f;
  constexpr float minimum = 0.08f;
  constexpr float maximum = 8.0f;
  std::vector<Candidate> candidates;
  for (int xi = 0; xi <= 25; ++xi) {
    const float x = -1.50f + 0.10f * xi;
    for (int yi = -6; yi <= 6; ++yi) {
      const float y = -1.40f + 0.10f * yi;
      for (int ai = -4; ai <= 4; ++ai) {
        const float yaw = 0.10f * ai;
        mj_resetData(model, data);
        data->qpos[qpos] = x;
        data->qpos[qpos + 1] = y;
        data->qpos[qpos + 2] = 0.68;
        data->qpos[qpos + 3] = std::cos(yaw * 0.5f);
        data->qpos[qpos + 4] = 0;
        data->qpos[qpos + 5] = 0;
        data->qpos[qpos + 6] = std::sin(yaw * 0.5f);
        mj_forward(model, data);
        Candidate candidate{x, y, yaw, std::vector<float>(horizontal * vertical,
          std::numeric_limits<float>::quiet_NaN())};
        const mjtNum* origin = data->cam_xpos + 3 * camera;
        const mjtNum* rotation = data->cam_xmat + 9 * camera;
        int index = 0;
        for (int row = 0; row < vertical; ++row) {
          const mjtNum pitch = (0.5 - static_cast<mjtNum>(row) / (vertical - 1)) * vertical_fov * mjPI / 180.0;
          for (int column = 0; column < horizontal; ++column, ++index) {
            const mjtNum ray_yaw = (0.5 - static_cast<mjtNum>(column) / (horizontal - 1)) * horizontal_fov * mjPI / 180.0;
            const mjtNum local[3] = {-std::sin(ray_yaw) * std::cos(pitch), std::sin(pitch), -std::cos(ray_yaw) * std::cos(pitch)};
            mjtNum world[3];
            mju_mulMatVec3(world, rotation, local);
            const mjtNum hit = RaySkippingRobot(model, data, origin, world, root, maximum);
            if (hit >= minimum && hit <= maximum) candidate.range[index] = static_cast<float>(hit);
          }
        }
        candidates.push_back(std::move(candidate));
      }
    }
  }
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  const char magic[8] = {'R','S','L','M','A','P','1','\0'};
  const std::uint32_t count = static_cast<std::uint32_t>(candidates.size());
  const std::uint32_t rays = horizontal * vertical;
  output.write(magic, sizeof(magic));
  output.write(reinterpret_cast<const char*>(&count), sizeof(count));
  output.write(reinterpret_cast<const char*>(&rays), sizeof(rays));
  for (const Candidate& candidate : candidates) {
    output.write(reinterpret_cast<const char*>(&candidate.x), sizeof(float));
    output.write(reinterpret_cast<const char*>(&candidate.y), sizeof(float));
    output.write(reinterpret_cast<const char*>(&candidate.yaw), sizeof(float));
    output.write(reinterpret_cast<const char*>(candidate.range.data()), sizeof(float) * candidate.range.size());
  }
  output.close();
  std::ofstream result(result_path);
  result << "{\"schema\":\"robotsim.issue80.lidar-static-map.v1\",\"status\":\"PASS\",\"candidates\":" << count << ",\"rays\":" << rays << ",\"x_range_m\":[-1.5,1.0],\"y_range_m\":[-2.0,-0.8],\"yaw_range_rad\":[-0.4,0.4],\"source\":\"STATIC_RAICOM_SCENE_MJ_RAY\",\"mj_step_calls\":0}\n";
  mj_deleteData(data); mj_deleteModel(model);
  return 0;
}
