// Engine-only validation for the Issue80 sensor plugin.  No ROS or robot model.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <mujoco/mujoco.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: engine_smoke <plugin_dir> <output_json>\n";
    return 2;
  }
  const std::string output_path(argv[2]);
  mj_loadAllPluginLibraries(argv[1], nullptr);
  const char* xml = R"(
<mujoco model="issue80_lidar_smoke">
  <extension><plugin plugin="robotsim.issue80.ray_caster_lidar"><instance name="lidar_instance">
    <config key="fov_h" value="90"/><config key="fov_v" value="20"/>
    <config key="size" value="9 3"/><config key="dis_range" value="0.08 8"/>
    <config key="n_step_update" value="1"/><config key="sensor_data_types" value="pos_w data"/>
  </instance></plugin></extension>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="sensor_mount" pos="0 0 1"><camera name="RayCasterLidar" mode="fixed" xyaxes="0 -1 0 0 0 1"/></body>
    <body pos="3 0 1"><geom type="box" size="0.1 1 1"/></body>
  </worldbody>
  <sensor>
    <plugin name="raycaster_lidar" instance="lidar_instance" objtype="camera" objname="RayCasterLidar"/>
  </sensor>
</mujoco>)";
  const std::string xml_path = output_path + ".xml";
  { std::ofstream stream(xml_path); stream << xml; }
  char error[1024] = {};
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  if (!model) {
    std::cerr << error << "\n";
    return 3;
  }
  mjData* data = mj_makeData(model);
  mj_step(model, data);
  const int sensor_id = mj_name2id(model, mjOBJ_SENSOR, "raycaster_lidar");
  const int address = sensor_id >= 0 ? model->sensor_adr[sensor_id] : -1;
  const int size = sensor_id >= 0 ? model->sensor_dim[sensor_id] : 0;
  int finite_xyz = 0;
  for (int index = 0; index + 2 < size / 4 * 3; index += 3) {
    const mjtNum* xyz = data->sensordata + address + index;
    if (std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2])) ++finite_xyz;
  }
  std::ofstream output(output_path);
  output << "{\"sensor_id\":" << sensor_id << ",\"sensor_dim\":" << size
         << ",\"finite_xyz_triplets\":" << finite_xyz << "}\n";
  const bool ok = sensor_id >= 0 && size == 9 * 3 * 4 && finite_xyz > 0;
  mj_deleteData(data);
  mj_deleteModel(model);
  return ok ? 0 : 4;
}
