// Read-only native MuJoCo state capture for the bounded Issue #80 standing
// diagnostic. This library is loaded only through a staged evidence launch
// script. It neither changes controls nor calls any MuJoCo stepping API.
//
// The build wrapper obtains the exact public MuJoCo 3.3.7 headers from the
// official tagged source, while the hook checks mj_versionString() again at
// runtime before it dereferences mjModel or mjData.

#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mujoco/mujoco.h>

#if mjVERSION_HEADER != 337
#error "This capture hook must be built with the exact MuJoCo 3.3.7 header."
#endif

typedef void (*mj_step_fn)(const mjModel* m, mjData* d);
typedef void (*mj_forward_fn)(const mjModel* m, mjData* d);
typedef void (*mj_contact_force_fn)(const mjModel* m, const mjData* d,
                                    int id, mjtNum result[6]);
typedef void (*mj_object_velocity_fn)(const mjModel* m, const mjData* d,
                                      int objtype, int objid,
                                      mjtNum res[6], int flg_local);
typedef const char* (*mj_version_string_fn)(void);
typedef int (*mj_name2id_fn)(const mjModel* m, int type, const char* name);
typedef const char* (*mj_id2name_fn)(const mjModel* m, int type, int id);

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t output_lock = PTHREAD_MUTEX_INITIALIZER;

static mj_step_fn real_mj_step = NULL;
static mj_forward_fn real_mj_forward = NULL;
static mj_contact_force_fn real_mj_contact_force = NULL;
static mj_object_velocity_fn real_mj_object_velocity = NULL;
static mj_version_string_fn real_mj_version_string = NULL;
static mj_name2id_fn real_mj_name2id = NULL;
static mj_id2name_fn real_mj_id2name = NULL;

static FILE* trace_stream = NULL;
static int capture_enabled = 0;
static int capture_error = 0;
static int metadata_written = 0;
static int step_calls = 0;
static int emitted_sample_count = 0;
static int sample_interval_steps = 10;
static int generation = 0;
static double last_observed_sim_time = NAN;
static double last_sample_sim_time = NAN;
static double last_sample_pelvis_xy[2] = {NAN, NAN};
static int last_sample_generation = -1;
static const mjModel* captured_model = NULL;
static int pelvis_body_id = -1;
static int left_foot_body_id = -1;
static int right_foot_body_id = -1;
static const char* pelvis_body_name = "pelvis";
static const char* left_foot_body_name = "left_ankle_roll_link";
static const char* right_foot_body_name = "right_ankle_roll_link";

static uint64_t monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
static const char* nonempty_env(const char* key, const char* fallback) {
  const char* value = getenv(key);
  return value && value[0] ? value : fallback;
}

static void json_string(FILE* stream, const char* text) {
  if (!text) {
    fputs("null", stream);
    return;
  }
  fputc('"', stream);
  for (const unsigned char* p = (const unsigned char*)text; *p; ++p) {
    switch (*p) {
      case '"': fputs("\\\"", stream); break;
      case '\\': fputs("\\\\", stream); break;
      case '\b': fputs("\\b", stream); break;
      case '\f': fputs("\\f", stream); break;
      case '\n': fputs("\\n", stream); break;
      case '\r': fputs("\\r", stream); break;
      case '\t': fputs("\\t", stream); break;
      default:
        if (*p < 0x20) {
          fprintf(stream, "\\u%04x", *p);
        } else {
          fputc(*p, stream);
        }
    }
  }
  fputc('"', stream);
}

static void json_num(FILE* stream, double value) {
  if (isfinite(value)) {
    fprintf(stream, "%.17g", value);
  } else {
    fputs("null", stream);
  }
}

static void json_num_array(FILE* stream, const mjtNum* values, int count) {
  fputc('[', stream);
  for (int i = 0; i < count; ++i) {
    if (i) fputc(',', stream);
    json_num(stream, values[i]);
  }
  fputc(']', stream);
}

static void json_vec3(FILE* stream, const mjtNum* values) {
  json_num_array(stream, values, 3);
}

static void json_vec6(FILE* stream, const mjtNum* values) {
  json_num_array(stream, values, 6);
}

static void capture_error_event(const char* code, const char* detail) {
  const char* path = getenv("ROBOTSIM_MJDATA_CAPTURE_PATH");
  if (!path || !path[0]) {
    return;
  }
  FILE* stream = trace_stream ? trace_stream : fopen(path, "a");
  if (!stream) {
    return;
  }
  fprintf(stream,
          "{\"schema\":\"robotsim.issue80.native-mjdata-capture.v1\","
          "\"event\":\"capture_error\",\"host_monotonic_ns\":%" PRIu64 ","
          "\"code\":",
          monotonic_ns());
  json_string(stream, code);
  fputs(",\"detail\":", stream);
  json_string(stream, detail);
  fputs("}\n", stream);
  fflush(stream);
  if (!trace_stream) fclose(stream);
}

static void* next_symbol(const char* name) {
  dlerror();
  void* value = dlsym(RTLD_NEXT, name);
  const char* error = dlerror();
  if (error || !value) {
    capture_error_event("DLSYM_FAILURE", error ? error : name);
    return NULL;
  }
  return value;
}

static void initialize_capture(void) {
  real_mj_step = (mj_step_fn)next_symbol("mj_step");
  real_mj_forward = (mj_forward_fn)next_symbol("mj_forward");
  real_mj_contact_force = (mj_contact_force_fn)next_symbol("mj_contactForce");
  real_mj_object_velocity = (mj_object_velocity_fn)next_symbol("mj_objectVelocity");
  real_mj_version_string = (mj_version_string_fn)next_symbol("mj_versionString");
  real_mj_name2id = (mj_name2id_fn)next_symbol("mj_name2id");
  real_mj_id2name = (mj_id2name_fn)next_symbol("mj_id2name");
  if (!real_mj_step || !real_mj_forward || !real_mj_contact_force ||
      !real_mj_object_velocity || !real_mj_version_string ||
      !real_mj_name2id || !real_mj_id2name) {
    capture_error = 1;
    return;
  }

  const char* output_path = getenv("ROBOTSIM_MJDATA_CAPTURE_PATH");
  if (!output_path || !output_path[0]) {
    return;
  }

  char* end = NULL;
  const char* interval = nonempty_env("ROBOTSIM_MJDATA_CAPTURE_INTERVAL_STEPS", "10");
  long parsed_interval = strtol(interval, &end, 10);
  if (!end || *end || parsed_interval < 1 || parsed_interval > 1000) {
    capture_error_event("INVALID_CAPTURE_INTERVAL", interval);
    capture_error = 1;
    return;
  }
  sample_interval_steps = (int)parsed_interval;

  const char* expected = nonempty_env("ROBOTSIM_MJDATA_CAPTURE_EXPECTED_VERSION", "3.3.7");
  const char* actual = real_mj_version_string();
  if (!actual || strcmp(actual, expected) != 0) {
    capture_error_event("MUJOCO_VERSION_MISMATCH", actual ? actual : "(null)");
    capture_error = 1;
    return;
  }

  trace_stream = fopen(output_path, "a");
  if (!trace_stream) {
    capture_error_event("TRACE_OPEN_FAILURE", strerror(errno));
    capture_error = 1;
    return;
  }
  setvbuf(trace_stream, NULL, _IOFBF, 1 << 16);
  pelvis_body_name = nonempty_env("ROBOTSIM_MJDATA_CAPTURE_PELVIS_BODY", "pelvis");
  left_foot_body_name = nonempty_env(
      "ROBOTSIM_MJDATA_CAPTURE_LEFT_FOOT_BODY", "left_ankle_roll_link");
  right_foot_body_name = nonempty_env(
      "ROBOTSIM_MJDATA_CAPTURE_RIGHT_FOOT_BODY", "right_ankle_roll_link");
  capture_enabled = 1;
}

static int reset_detected(const mjModel* m, const mjData* d) {
  const double current = d->time;
  // mj_step can call mj_forward at an intermediate zero-time stage while the
  // first step is still being assembled. Treat a rollback as a Reset only
  // after the simulation has advanced beyond an initial transient and by more
  // than one nominal step.
  const double meaningful_prior_time = fmax(0.1, 4.0 * m->opt.timestep);
  const int reset = isfinite(last_observed_sim_time) &&
                    last_observed_sim_time > meaningful_prior_time &&
                    current + 0.5 * m->opt.timestep < last_observed_sim_time;
  if (reset) {
    ++generation;
    last_sample_sim_time = NAN;
    last_sample_pelvis_xy[0] = NAN;
    last_sample_pelvis_xy[1] = NAN;
    last_sample_generation = -1;
  }
  last_observed_sim_time = current;
  return reset;
}

static void write_id_name(FILE* stream, const mjModel* m, int type, int id) {
  fprintf(stream, "{\"id\":%d,\"name\":", id);
  json_string(stream, id >= 0 ? real_mj_id2name(m, type, id) : NULL);
  fputc('}', stream);
}

static void write_model_metadata(FILE* stream, const mjModel* m, const mjData* d) {
  fprintf(stream,
          "{\"schema\":\"robotsim.issue80.native-mjdata-capture.v1\","
          "\"event\":\"model_metadata\",\"host_monotonic_ns\":%" PRIu64 ","
          "\"sim_time_s\":",
          monotonic_ns());
  json_num(stream, d->time);
  fprintf(stream,
          ",\"mujoco_runtime_version\":");
  json_string(stream, real_mj_version_string());
  fprintf(stream,
          ",\"generation\":%d,\"nq\":%d,\"nv\":%d,\"nu\":%d,"
          "\"nbody\":%d,\"ngeom\":%d,\"njnt\":%d,\"timestep_s\":",
          generation, m->nq, m->nv, m->nu, m->nbody, m->ngeom, m->njnt);
  json_num(stream, m->opt.timestep);
  fputs(",\"capture_interval_steps\":", stream);
  fprintf(stream, "%d", sample_interval_steps);
  fputs(",\"body_names\":[", stream);
  for (int i = 0; i < m->nbody; ++i) {
    if (i) fputc(',', stream);
    write_id_name(stream, m, mjOBJ_BODY, i);
  }
  fputs("],\"geom_names\":[", stream);
  for (int i = 0; i < m->ngeom; ++i) {
    if (i) fputc(',', stream);
    fprintf(stream, "{\"id\":%d,\"name\":", i);
    json_string(stream, real_mj_id2name(m, mjOBJ_GEOM, i));
    fprintf(stream, ",\"body_id\":%d}", m->geom_bodyid[i]);
  }
  fputs("],\"joint_names\":[", stream);
  for (int i = 0; i < m->njnt; ++i) {
    if (i) fputc(',', stream);
    fprintf(stream,
            "{\"id\":%d,\"name\":", i);
    json_string(stream, real_mj_id2name(m, mjOBJ_JOINT, i));
    fprintf(stream,
            ",\"type\":%d,\"qpos_adr\":%d,\"dof_adr\":%d}",
            m->jnt_type[i], m->jnt_qposadr[i], m->jnt_dofadr[i]);
  }
  fputs("],\"actuator_names\":[", stream);
  for (int i = 0; i < m->nu; ++i) {
    if (i) fputc(',', stream);
    write_id_name(stream, m, mjOBJ_ACTUATOR, i);
  }
  fputs("],\"selected_bodies\":{\"pelvis\":", stream);
  write_id_name(stream, m, mjOBJ_BODY, pelvis_body_id);
  fputs(",\"left_foot\":", stream);
  write_id_name(stream, m, mjOBJ_BODY, left_foot_body_id);
  fputs(",\"right_foot\":", stream);
  write_id_name(stream, m, mjOBJ_BODY, right_foot_body_id);
  fputs("}}\n", stream);
  fflush(stream);
}

static void write_body_state(FILE* stream, const mjModel* m, const mjData* d,
                             const char* label, int body_id) {
  fprintf(stream, "\"%s\":{\"id\":%d,\"name\":", label, body_id);
  json_string(stream, body_id >= 0 ? real_mj_id2name(m, mjOBJ_BODY, body_id) : NULL);
  if (body_id < 0) {
    fputs("}", stream);
    return;
  }
  mjtNum velocity[6] = {0, 0, 0, 0, 0, 0};
  real_mj_object_velocity(m, d, mjOBJ_BODY, body_id, velocity, 0);
  fputs(",\"world_position_m\":", stream);
  json_vec3(stream, d->xpos + 3 * body_id);
  fputs(",\"world_quaternion_wxyz\":", stream);
  json_num_array(stream, d->xquat + 4 * body_id, 4);
  fputs(",\"world_spatial_velocity_rot_lin\":", stream);
  json_vec6(stream, velocity);
  fputs(",\"com_spatial_velocity_rot_lin\":", stream);
  json_vec6(stream, d->cvel + 6 * body_id);
  fputs("}", stream);
}

static double dot3(const mjtNum* first, const mjtNum* second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

static void cross3(const mjtNum* first, const mjtNum* second, mjtNum* result) {
  result[0] = first[1] * second[2] - first[2] * second[1];
  result[1] = first[2] * second[0] - first[0] * second[2];
  result[2] = first[0] * second[1] - first[1] * second[0];
}

static int geom_body_id(const mjModel* m, int geom_id) {
  return geom_id >= 0 && geom_id < m->ngeom ? m->geom_bodyid[geom_id] : -1;
}

static void write_contact(FILE* stream, const mjModel* m, const mjData* d,
                          int contact_id) {
  const mjContact* contact = d->contact + contact_id;
  const int geom1 = contact->geom[0] >= 0 ? contact->geom[0] : contact->geom1;
  const int geom2 = contact->geom[1] >= 0 ? contact->geom[1] : contact->geom2;
  const int body1 = geom_body_id(m, geom1);
  const int body2 = geom_body_id(m, geom2);
  mjtNum force[6] = {0, 0, 0, 0, 0, 0};
  if (contact->efc_address >= 0) {
    real_mj_contact_force(m, d, contact_id, force);
  }

  mjtNum spatial1[6] = {0, 0, 0, 0, 0, 0};
  mjtNum spatial2[6] = {0, 0, 0, 0, 0, 0};
  if (geom1 >= 0) real_mj_object_velocity(m, d, mjOBJ_GEOM, geom1, spatial1, 0);
  if (geom2 >= 0) real_mj_object_velocity(m, d, mjOBJ_GEOM, geom2, spatial2, 0);

  mjtNum at_contact1[3] = {spatial1[3], spatial1[4], spatial1[5]};
  mjtNum at_contact2[3] = {spatial2[3], spatial2[4], spatial2[5]};
  if (geom1 >= 0) {
    mjtNum arm[3] = {
      contact->pos[0] - d->geom_xpos[3 * geom1],
      contact->pos[1] - d->geom_xpos[3 * geom1],
      contact->pos[2] - d->geom_xpos[3 * geom1],
    };
    mjtNum correction[3];
    cross3(spatial1, arm, correction);
    for (int i = 0; i < 3; ++i) at_contact1[i] += correction[i];
  }
  if (geom2 >= 0) {
    mjtNum arm[3] = {
      contact->pos[0] - d->geom_xpos[3 * geom2],
      contact->pos[1] - d->geom_xpos[3 * geom2],
      contact->pos[2] - d->geom_xpos[3 * geom2],
    };
    mjtNum correction[3];
    cross3(spatial2, arm, correction);
    for (int i = 0; i < 3; ++i) at_contact2[i] += correction[i];
  }
  mjtNum relative[3] = {
    at_contact2[0] - at_contact1[0],
    at_contact2[1] - at_contact1[1],
    at_contact2[2] - at_contact1[2],
  };
  const mjtNum* normal = contact->frame;
  const mjtNum* tangent1 = contact->frame + 3;
  const mjtNum* tangent2 = contact->frame + 6;
  const double rel_normal = dot3(relative, normal);
  const double rel_tangent1 = dot3(relative, tangent1);
  const double rel_tangent2 = dot3(relative, tangent2);
  const double tangential_speed = hypot(rel_tangent1, rel_tangent2);
  mjtNum world_force[3] = {
    force[0] * normal[0] + force[1] * tangent1[0] + force[2] * tangent2[0],
    force[0] * normal[1] + force[1] * tangent1[1] + force[2] * tangent2[1],
    force[0] * normal[2] + force[1] * tangent1[2] + force[2] * tangent2[2],
  };

  const int next_address = contact_id + 1 < d->ncon
      ? d->contact[contact_id + 1].efc_address : d->nefc;
  const int efc_start = contact->efc_address;
  const int efc_end = efc_start >= 0 && next_address >= efc_start
      ? (next_address < d->nefc ? next_address : d->nefc) : efc_start;

  fprintf(stream,
          "{\"id\":%d,\"dim\":%d,\"distance_m\":",
          contact_id, contact->dim);
  json_num(stream, contact->dist);
  fputs(",\"position_world_m\":", stream);
  json_vec3(stream, contact->pos);
  fputs(",\"frame_world_normal_tangent1_tangent2\":", stream);
  json_num_array(stream, contact->frame, 9);
  fputs(",\"geom1\":{\"id\":", stream);
  fprintf(stream, "%d,\"name\":", geom1);
  json_string(stream, geom1 >= 0 ? real_mj_id2name(m, mjOBJ_GEOM, geom1) : NULL);
  fprintf(stream, ",\"body_id\":%d,\"body_name\":", body1);
  json_string(stream, body1 >= 0 ? real_mj_id2name(m, mjOBJ_BODY, body1) : NULL);
  fputs("},\"geom2\":{\"id\":", stream);
  fprintf(stream, "%d,\"name\":", geom2);
  json_string(stream, geom2 >= 0 ? real_mj_id2name(m, mjOBJ_GEOM, geom2) : NULL);
  fprintf(stream, ",\"body_id\":%d,\"body_name\":", body2);
  json_string(stream, body2 >= 0 ? real_mj_id2name(m, mjOBJ_BODY, body2) : NULL);
  fputs("},\"raw_contact_force_normal_tangent1_tangent2_torque\":", stream);
  json_vec6(stream, force);
  fputs(",\"contact_force_world_from_contact_frame\":", stream);
  json_vec3(stream, world_force);
  fputs(",\"normal_force_n\":", stream);
  json_num(stream, force[0]);
  fputs(",\"tangential_force_norm_n\":", stream);
  json_num(stream, hypot(force[1], force[2]));
  fputs(",\"force_times_timestep_impulse_Ns\":", stream);
  json_num(stream, force[0] * m->opt.timestep);
  fputs(",\"tangential_force_times_timestep_impulse_Ns\":", stream);
  json_num(stream, hypot(force[1], force[2]) * m->opt.timestep);
  fputs(",\"geom1_spatial_velocity_world_rot_lin\":", stream);
  json_vec6(stream, spatial1);
  fputs(",\"geom2_spatial_velocity_world_rot_lin\":", stream);
  json_vec6(stream, spatial2);
  fputs(",\"relative_contact_velocity_geom2_minus_geom1_world_m_s\":", stream);
  json_vec3(stream, relative);
  fputs(",\"relative_normal_velocity_m_s\":", stream);
  json_num(stream, rel_normal);
  fputs(",\"relative_tangent1_velocity_m_s\":", stream);
  json_num(stream, rel_tangent1);
  fputs(",\"relative_tangent2_velocity_m_s\":", stream);
  json_num(stream, rel_tangent2);
  fputs(",\"relative_tangential_speed_m_s\":", stream);
  json_num(stream, tangential_speed);
  fprintf(stream,
          ",\"involves_selected_left_foot\":%s,"
          "\"involves_selected_right_foot\":%s,"
          "\"efc_address\":%d,\"efc_velocity_raw\":[",
          body1 == left_foot_body_id || body2 == left_foot_body_id ? "true" : "false",
          body1 == right_foot_body_id || body2 == right_foot_body_id ? "true" : "false",
          efc_start);
  if (efc_start >= 0 && efc_end >= efc_start) {
    for (int i = efc_start; i < efc_end; ++i) {
      if (i > efc_start) fputc(',', stream);
      json_num(stream, d->efc_vel[i]);
    }
  }
  fputs("],\"efc_force_raw\":[", stream);
  if (efc_start >= 0 && efc_end >= efc_start) {
    for (int i = efc_start; i < efc_end; ++i) {
      if (i > efc_start) fputc(',', stream);
      json_num(stream, d->efc_force[i]);
    }
  }
  fputs("]}", stream);
}

static int has_floor_name(const mjModel* m, int geom_id) {
  const char* name = geom_id >= 0 ? real_mj_id2name(m, mjOBJ_GEOM, geom_id) : NULL;
  if (!name) return 0;
  for (const char* p = name; p[0] && p[1] && p[2] && p[3] && p[4]; ++p) {
    if (tolower((unsigned char)p[0]) == 'f' &&
        tolower((unsigned char)p[1]) == 'l' &&
        tolower((unsigned char)p[2]) == 'o' &&
        tolower((unsigned char)p[3]) == 'o' &&
        tolower((unsigned char)p[4]) == 'r') {
      return 1;
    }
  }
  return 0;
}

static int is_selected_foot_floor_contact(const mjModel* m, const mjData* d,
                                          int contact_id) {
  const mjContact* contact = d->contact + contact_id;
  const int geom1 = contact->geom[0] >= 0 ? contact->geom[0] : contact->geom1;
  const int geom2 = contact->geom[1] >= 0 ? contact->geom[1] : contact->geom2;
  const int body1 = geom_body_id(m, geom1);
  const int body2 = geom_body_id(m, geom2);
  const int selected_foot = body1 == left_foot_body_id ||
                            body1 == right_foot_body_id ||
                            body2 == left_foot_body_id ||
                            body2 == right_foot_body_id;
  return selected_foot && (has_floor_name(m, geom1) || has_floor_name(m, geom2));
}

static void capture_sample(const char* event, const mjModel* m, const mjData* d) {
  if (!capture_enabled || capture_error || !m || !d) return;
  if (pthread_mutex_trylock(&output_lock) != 0) return;
  if (!trace_stream) {
    pthread_mutex_unlock(&output_lock);
    return;
  }
  if (captured_model && captured_model != m) {
    capture_error_event("MULTIPLE_MJMODEL_POINTERS", "capture rejects an unexpected secondary model");
    capture_error = 1;
    pthread_mutex_unlock(&output_lock);
    return;
  }
  captured_model = m;
  if (!metadata_written) {
    pelvis_body_id = real_mj_name2id(m, mjOBJ_BODY, pelvis_body_name);
    left_foot_body_id = real_mj_name2id(m, mjOBJ_BODY, left_foot_body_name);
    right_foot_body_id = real_mj_name2id(m, mjOBJ_BODY, right_foot_body_name);
    if (pelvis_body_id < 0 || left_foot_body_id < 0 || right_foot_body_id < 0) {
      capture_error_event("REQUIRED_BODY_NAME_NOT_FOUND", "pelvis or ankle-roll link lookup failed");
      capture_error = 1;
      pthread_mutex_unlock(&output_lock);
      return;
    }
    write_model_metadata(trace_stream, m, d);
    metadata_written = 1;
  }

  const mjtNum* pelvis_position = d->xpos + 3 * pelvis_body_id;
  const double sample_time = d->time;
  double fd_velocity[2] = {NAN, NAN};
  double fd_displacement[2] = {NAN, NAN};
  if (last_sample_generation == generation && isfinite(last_sample_sim_time) &&
      sample_time > last_sample_sim_time + 1e-12) {
    const double dt = sample_time - last_sample_sim_time;
    fd_displacement[0] = pelvis_position[0] - last_sample_pelvis_xy[0];
    fd_displacement[1] = pelvis_position[1] - last_sample_pelvis_xy[1];
    fd_velocity[0] = fd_displacement[0] / dt;
    fd_velocity[1] = fd_displacement[1] / dt;
  }

  fprintf(trace_stream,
          "{\"schema\":\"robotsim.issue80.native-mjdata-capture.v1\","
          "\"event\":");
  json_string(trace_stream, event);
  fprintf(trace_stream,
          ",\"host_monotonic_ns\":%" PRIu64 ",\"sim_time_s\":",
          monotonic_ns());
  json_num(trace_stream, sample_time);
  fprintf(trace_stream,
          ",\"generation\":%d,\"mj_step_call_index\":%d,"
          "\"timestep_s\":",
          generation, step_calls);
  json_num(trace_stream, m->opt.timestep);
  fputs(",\"mujoco_timer\":{\"step_duration_raw\":", trace_stream);
  json_num(trace_stream, d->timer[mjTIMER_STEP].duration);
  fprintf(trace_stream, ",\"step_number\":%d,\"forward_duration_raw\":",
          d->timer[mjTIMER_STEP].number);
  json_num(trace_stream, d->timer[mjTIMER_FORWARD].duration);
  fprintf(trace_stream, ",\"forward_number\":%d}",
          d->timer[mjTIMER_FORWARD].number);
  fputs(",\"pelvis\":{", trace_stream);
  write_body_state(trace_stream, m, d, "state", pelvis_body_id);
  fputs("},\"feet\":{", trace_stream);
  write_body_state(trace_stream, m, d, "left", left_foot_body_id);
  fputc(',', trace_stream);
  write_body_state(trace_stream, m, d, "right", right_foot_body_id);
  fputs("},\"pelvis_finite_difference_xy\":{"
       "\"delta_world_xy_m\":[", trace_stream);
  json_num(trace_stream, fd_displacement[0]);
  fputc(',', trace_stream);
  json_num(trace_stream, fd_displacement[1]);
  fputs("],\"velocity_world_xy_m_s\":[", trace_stream);
  json_num(trace_stream, fd_velocity[0]);
  fputc(',', trace_stream);
  json_num(trace_stream, fd_velocity[1]);
  fputs("]},\"qpos\":", trace_stream);
  json_num_array(trace_stream, d->qpos, m->nq);
  fputs(",\"qvel\":", trace_stream);
  json_num_array(trace_stream, d->qvel, m->nv);
  fputs(",\"ctrl\":", trace_stream);
  json_num_array(trace_stream, d->ctrl, m->nu);
  fputs(",\"actuator_force\":", trace_stream);
  json_num_array(trace_stream, d->actuator_force, m->nu);
  fputs(",\"qfrc_actuator\":", trace_stream);
  json_num_array(trace_stream, d->qfrc_actuator, m->nv);
  int selected_foot_floor_contacts = 0;
  for (int i = 0; i < d->ncon; ++i) {
    selected_foot_floor_contacts += is_selected_foot_floor_contact(m, d, i);
  }
  fprintf(trace_stream,
          ",\"contact_count\":%d,\"captured_foot_floor_contact_count\":%d,"
          "\"contacts\":[",
          d->ncon, selected_foot_floor_contacts);
  int written_contacts = 0;
  for (int i = 0; i < d->ncon; ++i) {
    if (!is_selected_foot_floor_contact(m, d, i)) continue;
    if (written_contacts++) fputc(',', trace_stream);
    write_contact(trace_stream, m, d, i);
  }
  fputs("]}\n", trace_stream);
  ++emitted_sample_count;
  if (strcmp(event, "post_step") != 0 || emitted_sample_count % 10 == 0) {
    fflush(trace_stream);
  }

  last_sample_sim_time = sample_time;
  last_sample_pelvis_xy[0] = pelvis_position[0];
  last_sample_pelvis_xy[1] = pelvis_position[1];
  last_sample_generation = generation;
  pthread_mutex_unlock(&output_lock);
}

void mj_step(const mjModel* m, mjData* d) {
  pthread_once(&init_once, initialize_capture);
  if (!real_mj_step) {
    capture_error_event("REAL_MJ_STEP_UNAVAILABLE", "original mj_step could not be resolved");
    return;
  }
  real_mj_step(m, d);
  ++step_calls;
  const int reset = reset_detected(m, d);
  if (capture_enabled && (reset || step_calls % sample_interval_steps == 0)) {
    capture_sample(reset ? "post_step_after_reset_generation" : "post_step", m, d);
  }
}

void mj_forward(const mjModel* m, mjData* d) {
  pthread_once(&init_once, initialize_capture);
  if (!real_mj_forward) {
    capture_error_event("REAL_MJ_FORWARD_UNAVAILABLE", "original mj_forward could not be resolved");
    return;
  }
  real_mj_forward(m, d);
  const int reset = reset_detected(m, d);
  if (capture_enabled && (!metadata_written || reset)) {
    capture_sample(reset ? "post_forward_reset_generation" : "post_forward_initial", m, d);
  }
}
