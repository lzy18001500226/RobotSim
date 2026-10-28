// Minimal SIMULATION_ONLY RayCasterLidar plugin for the delivered RAICOM runtime.
//
// It emits world-frame ray-hit XYZ values followed by ray distance.  The first
// six plugin-state values document that layout for the precompiled publisher:
// [horizontal, vertical, xyz_offset, xyz_length, distance_offset, distance_length].
// This is a sensor-only extension: it never writes qpos, actuator controls,
// contacts, robot geometry, or model parameters.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

namespace {

struct LidarState {
  int horizontal = 0;
  int vertical = 0;
  int update_every = 1;
  int steps = 0;
  int sensor_id = -1;
  int camera_id = -1;
  int robot_root_body = -1;
  mjtNum fov_horizontal = 0;
  mjtNum fov_vertical = 0;
  mjtNum min_range = 0;
  mjtNum max_range = 0;
  std::ofstream raw_trace;
  unsigned long long raw_sequence = 0;
};

std::vector<mjtNum> Parse(const char* text) {
  std::vector<mjtNum> result;
  std::istringstream stream(text ? text : "");
  mjtNum value = 0;
  while (stream >> value) result.push_back(value);
  return result;
}

int FindSensor(const mjModel* model, int instance) {
  for (int id = 0; id < model->nsensor; ++id) {
    // During plugin init, the delivery's MuJoCo 3.3.7 loader has already
    // assigned sensor_plugin but has not yet finalized sensor_type.  The
    // instance mapping is the stable lookup key at this callback stage.
    if (model->sensor_plugin[id] == instance) {
      return id;
    }
  }
  return -1;
}

int RootBody(const mjModel* model, int body_id) {
  while (body_id > 0 && model->body_parentid[body_id] > 0) body_id = model->body_parentid[body_id];
  return body_id;
}

mjtNum RaySkippingRobot(const mjModel* model, const mjData* data,
                        const mjtNum origin[3], const mjtNum direction[3],
                        int robot_root_body, mjtNum max_range, int* geom_id) {
  mjtNum shifted_origin[3] = {origin[0], origin[1], origin[2]};
  mjtNum travelled = 0;
  for (int attempt = 0; attempt < 64 && travelled < max_range; ++attempt) {
    int hit_geom = -1;
    const mjtNum hit = mj_ray(model, data, shifted_origin, direction, nullptr, 1, -1, &hit_geom);
    if (hit < 0 || hit_geom < 0) {
      *geom_id = -1;
      return -1;
    }
    if (RootBody(model, model->geom_bodyid[hit_geom]) != robot_root_body) {
      *geom_id = hit_geom;
      return travelled + hit;
    }
    const mjtNum advance = hit + 1e-3;
    travelled += advance;
    for (int axis = 0; axis < 3; ++axis) shifted_origin[axis] += advance * direction[axis];
  }
  *geom_id = -1;
  return -1;
}

void WriteMetadata(mjtNum* state, const LidarState* lidar) {
  const int rays = lidar->horizontal * lidar->vertical;
  state[0] = lidar->horizontal;
  state[1] = lidar->vertical;
  state[2] = 0;
  state[3] = rays * 3;
  state[4] = rays * 3;
  state[5] = rays;
}

int RaySensorSize(const mjModel* model, int instance, int) {
  const auto size = Parse(mj_getPluginConfig(model, instance, "size"));
  if (size.size() != 2 || size[0] <= 0 || size[1] <= 0) {
    std::fprintf(stderr, "issue80_lidar_size invalid instance=%d size_count=%zu\n", instance, size.size());
    mju_error("Issue80 lidar requires positive size H V");
  }
  const int dimension = static_cast<int>(size[0] * size[1] * 4);
  std::fprintf(stderr, "issue80_lidar_size instance=%d H=%.0f V=%.0f dim=%d\n",
               instance, size[0], size[1], dimension);
  return dimension;
}

int Init(const mjModel* model, mjData* data, int instance) {
  const auto size = Parse(mj_getPluginConfig(model, instance, "size"));
  const auto fov_h = Parse(mj_getPluginConfig(model, instance, "fov_h"));
  const auto fov_v = Parse(mj_getPluginConfig(model, instance, "fov_v"));
  const auto range = Parse(mj_getPluginConfig(model, instance, "dis_range"));
  const auto cadence = Parse(mj_getPluginConfig(model, instance, "n_step_update"));
  if (size.size() != 2 || fov_h.size() != 1 || fov_v.size() != 1 || range.size() != 2) {
    std::fprintf(stderr, "issue80_lidar_init invalid config sizes: size=%zu fov_h=%zu fov_v=%zu range=%zu\n",
                 size.size(), fov_h.size(), fov_v.size(), range.size());
    return -1;
  }
  auto* lidar = new LidarState;
  lidar->horizontal = static_cast<int>(size[0]);
  lidar->vertical = static_cast<int>(size[1]);
  lidar->fov_horizontal = fov_h[0] * mjPI / 180.0;
  lidar->fov_vertical = fov_v[0] * mjPI / 180.0;
  lidar->min_range = range[0];
  lidar->max_range = range[1];
  if (!(lidar->min_range > 0 && lidar->max_range > lidar->min_range)) {
    std::fprintf(stderr, "issue80_lidar_init invalid range: %.6f %.6f\n", lidar->min_range, lidar->max_range);
    delete lidar;
    return -1;
  }
  if (!cadence.empty()) lidar->update_every = std::max(1, static_cast<int>(cadence[0]));
  lidar->sensor_id = FindSensor(model, instance);
  if (lidar->sensor_id < 0) {
    std::fprintf(stderr, "issue80_lidar_init sensor not found: instance=%d nplugin=%d nsensor=%d",
                 instance, model->nplugin, model->nsensor);
    for (int id = 0; id < model->nsensor; ++id) {
      std::fprintf(stderr, " [sensor=%d type=%d plugin=%d]", id, model->sensor_type[id], model->sensor_plugin[id]);
    }
    std::fprintf(stderr, "\n");
    delete lidar;
    return -1;
  }
  lidar->camera_id = model->sensor_objid[lidar->sensor_id];
  if (lidar->camera_id < 0 || lidar->camera_id >= model->ncam) {
    std::fprintf(stderr, "issue80_lidar_init invalid camera: sensor=%d camera=%d ncam=%d\n",
                 lidar->sensor_id, lidar->camera_id, model->ncam);
    delete lidar;
    return -1;
  }
  lidar->robot_root_body = RootBody(model, model->cam_bodyid[lidar->camera_id]);
  if (const char* trace_path = std::getenv("ROBOTSIM_LIDAR_RAW_TRACE")) {
    if (trace_path[0]) {
      lidar->raw_trace.open(trace_path, std::ios::out | std::ios::app);
      if (!lidar->raw_trace) {
        std::fprintf(stderr, "issue80_lidar_init raw trace open failed: %s\n", trace_path);
        delete lidar;
        return -1;
      }
    }
  }
  std::fprintf(stderr, "issue80_lidar_init ok sensor=%d camera=%d body=%d root=%d\n",
               lidar->sensor_id, lidar->camera_id, model->cam_bodyid[lidar->camera_id], lidar->robot_root_body);
  data->plugin_data[instance] = reinterpret_cast<uintptr_t>(lidar);
  WriteMetadata(data->plugin_state + model->plugin_stateadr[instance], lidar);
  return 0;
}

void Reset(const mjModel*, mjtNum* state, void* plugin_data, int) {
  auto* lidar = reinterpret_cast<LidarState*>(plugin_data);
  lidar->steps = 0;
  WriteMetadata(state, lidar);
}

void Destroy(mjData* data, int instance) {
  delete reinterpret_cast<LidarState*>(data->plugin_data[instance]);
  data->plugin_data[instance] = 0;
}

void Compute(const mjModel* model, mjData* data, int instance, int) {
  auto* lidar = reinterpret_cast<LidarState*>(data->plugin_data[instance]);
  if (!lidar || ++lidar->steps < lidar->update_every) return;
  lidar->steps = 0;
  WriteMetadata(data->plugin_state + model->plugin_stateadr[instance], lidar);

  const int rays = lidar->horizontal * lidar->vertical;
  mjtNum* xyz = data->sensordata + model->sensor_adr[lidar->sensor_id];
  mjtNum* distance = xyz + rays * 3;
  const mjtNum* origin = data->cam_xpos + 3 * lidar->camera_id;
  const mjtNum* rotation = data->cam_xmat + 9 * lidar->camera_id;
  const mjtNum nan = std::numeric_limits<mjtNum>::quiet_NaN();
  int ray_index = 0;
  for (int vertical = 0; vertical < lidar->vertical; ++vertical) {
    const mjtNum pitch = lidar->vertical == 1 ? 0 :
        (0.5 - static_cast<mjtNum>(vertical) / (lidar->vertical - 1)) * lidar->fov_vertical;
    for (int horizontal = 0; horizontal < lidar->horizontal; ++horizontal, ++ray_index) {
      const mjtNum yaw = lidar->horizontal == 1 ? 0 :
          (0.5 - static_cast<mjtNum>(horizontal) / (lidar->horizontal - 1)) * lidar->fov_horizontal;
      const mjtNum local_direction[3] = {
          -std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
      mjtNum world_direction[3];
      mju_mulMatVec3(world_direction, rotation, local_direction);
      int geom_id = -1;
      const mjtNum hit = RaySkippingRobot(model, data, origin, world_direction,
                                           lidar->robot_root_body, lidar->max_range, &geom_id);
      mjtNum* point = xyz + ray_index * 3;
      if (hit >= lidar->min_range && hit <= lidar->max_range && geom_id >= 0) {
        // Publish LiDAR-frame coordinates.  The delivered ``pos_b``
        // conversion publishes the resulting PointCloud2 in map frame; this
        // is retained as a precompiled-runtime limitation.
        point[0] = hit * local_direction[0];
        point[1] = hit * local_direction[1];
        point[2] = hit * local_direction[2];
        distance[ray_index] = hit;
      } else {
        point[0] = nan;
        point[1] = nan;
        point[2] = nan;
        distance[ray_index] = lidar->max_range;
      }
    }
  }
  // The delivered publisher converts pos_b to map-frame coordinates.  Retain
  // this plugin's source-local samples separately for the SIMULATION_ONLY ROS
  // bridge; this never feeds MC and never writes MuJoCo state.
  if (lidar->raw_trace.is_open()) {
    lidar->raw_trace << "{\"schema\":\"robotsim.issue80.lidar-raw.v1\","
                     << "\"sequence\":" << lidar->raw_sequence++ << ","
                     << "\"horizontal\":" << lidar->horizontal << ","
                     << "\"vertical\":" << lidar->vertical << ",\"xyz\":[";
    for (int index = 0; index < rays; ++index) {
      if (index) lidar->raw_trace << ',';
      const mjtNum* point = xyz + index * 3;
      if (std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2])) {
        lidar->raw_trace << '[' << point[0] << ',' << point[1] << ',' << point[2] << ']';
      } else {
        lidar->raw_trace << "null";
      }
    }
    lidar->raw_trace << "]}\n";
    lidar->raw_trace.flush();
  }
}

constexpr const char* kAttributes[] = {
    "fov_h", "fov_v", "size", "dis_range", "n_step_update", "sensor_data_types"};

}  // namespace

mjPLUGIN_LIB_INIT {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "robotsim.issue80.ray_caster_lidar";
  plugin.nattribute = sizeof(kAttributes) / sizeof(kAttributes[0]);
  plugin.attributes = kAttributes;
  plugin.capabilityflags = mjPLUGIN_SENSOR;
  plugin.needstage = mjSTAGE_POS;
  plugin.nstate = +[](const mjModel*, int) { return 6; };
  plugin.nsensordata = RaySensorSize;
  plugin.init = Init;
  plugin.destroy = Destroy;
  plugin.reset = Reset;
  plugin.compute = Compute;
  mjp_registerPlugin(&plugin);
}
