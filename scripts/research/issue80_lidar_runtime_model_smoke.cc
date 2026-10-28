// Load one generated LiDAR overlay through the delivered MuJoCo runtime.
// This helper never starts the vendor simulator or MC and never calls mj_step.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

static void JsonString(std::ostream& stream, const std::string& value) {
  stream << '"';
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') stream << '\\';
    stream << ch;
  }
  stream << '"';
}

static std::vector<std::string> g_loaded_plugin_libraries;

static void RecordPluginLibrary(const char* filename, int, int) {
  if (filename) g_loaded_plugin_libraries.emplace_back(filename);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: lidar_runtime_model_smoke <plugin_dir> <scene_xml> <output_json>\n";
    return 2;
  }
  const std::string plugin_dir(argv[1]);
  const std::string scene_path(argv[2]);
  const std::string output_path(argv[3]);
  mj_loadAllPluginLibraries(plugin_dir.c_str(), RecordPluginLibrary);
  char xml_error[4096] = {};
  mjModel* model = mj_loadXML(scene_path.c_str(), nullptr, xml_error, sizeof(xml_error));
  std::ofstream output(output_path);
  output << "{\"schema\":\"robotsim.issue80.lidar-runtime-model-smoke.v1\","
         << "\"mujoco_version\":";
  JsonString(output, mj_versionString());
  output << ",\"loaded_plugin_libraries\":[";
  for (std::size_t index = 0; index < g_loaded_plugin_libraries.size(); ++index) {
    if (index) output << ',';
    JsonString(output, g_loaded_plugin_libraries[index]);
  }
  output << ']';
  output << ",\"scene\":";
  JsonString(output, scene_path);
  if (!model) {
    output << ",\"status\":\"FAIL_MODEL_LOAD\",\"model_error\":";
    JsonString(output, xml_error);
    output << "}\n";
    return 3;
  }
  output << ",\"status\":\"PASS_MODEL_LOAD\","
         << "\"nbody\":" << model->nbody << ','
         << "\"njnt\":" << model->njnt << ','
         << "\"nu\":" << model->nu << ','
         << "\"nsensor\":" << model->nsensor << "}\n";
  mj_deleteModel(model);
  return 0;
}
