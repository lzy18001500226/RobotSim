/* Read-only mj_step observer. Build against the exact installed engine ABI.
 * Interposes the public function, calls it exactly once, then samples state.
 * No model, qpos, qvel, ctrl, force or contact fields are written here.
 */
#define _GNU_SOURCE
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

static pthread_once_t once = PTHREAD_ONCE_INIT;
static void (*step_real)(const mjModel*, mjData*);
static int (*version_real)(void);
static int (*id_real)(const mjModel*, int, const char*);
static const char* (*name_real)(const mjModel*, int, int);
static void (*force_real)(const mjModel*, const mjData*, int, mjtNum*);
static void (*velocity_real)(const mjModel*, const mjData*, int, int, mjtNum*, int);
static FILE* sink;
static char latest_path[4096], latest_tmp[4104];
static int count, generation;
static double previous = -1, next_time = -1;
static mjtNum peak_qvel[4096];
static double qvel_window_start;
static long long last_step_end_ns,max_call_gap_ns,max_step_duration_ns,last_io_ns,max_io_ns;

static long long mono_ns(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (long long)t.tv_sec*1000000000LL + t.tv_nsec;
}
static void init(void) {
  step_real=dlsym(RTLD_NEXT,"mj_step");
  version_real=dlsym(RTLD_NEXT,"mj_version");
  id_real=dlsym(RTLD_NEXT,"mj_name2id");
  name_real=dlsym(RTLD_NEXT,"mj_id2name");
  force_real=dlsym(RTLD_NEXT,"mj_contactForce");
  velocity_real=dlsym(RTLD_NEXT,"mj_objectVelocity");
  if (!step_real) _exit(125);
  /* Fail measurement closed on ABI mismatch; never dereference mjData then. */
  if (!version_real || version_real()!=mjVERSION_HEADER || !id_real || !name_real || !force_real || !velocity_real) {
    fprintf(stderr,"NATIVE_GRASP_AUDIT_ABI_OR_API_MISMATCH\n"); return;
  }
  const char* path=getenv("ROBOTSIM_NATIVE_AUDIT_OUT");
  if (path && *path) {
    sink=fopen(path,"wx");
    const char* slash=strrchr(path,'/');
    if(slash) {
      snprintf(latest_path,sizeof(latest_path),"%.*s/native_latest.json",(int)(slash-path),path);
      snprintf(latest_tmp,sizeof(latest_tmp),"%s.tmp",latest_path);
    }
  }
  if (!sink) fprintf(stderr,"NATIVE_GRASP_AUDIT_NO_EXCLUSIVE_OUTPUT\n");
}
static void array(const mjtNum* x, int n) {
  fputc('[',sink);
  for (int i=0;i<n;i++) fprintf(sink,"%s%.17g",i?",":"",(double)x[i]);
  fputc(']',sink);
}
static void body(const mjModel* m,const mjData* d,const char* name) {
  int b=id_real(m,mjOBJ_BODY,name);
  fprintf(sink,"\"%s\":",name);
  if (b<0) {fputs("null",sink); return;}
  fputs("{\"position_m\":",sink);array(d->xpos+3*b,3);
  fputs(",\"quaternion_wxyz\":",sink);array(d->xquat+4*b,4);
  fputs(",\"com_m\":",sink);array(d->xipos+3*b,3);
  fprintf(sink,",\"mass_kg\":%.17g",(double)m->body_mass[b]);
  fputs(",\"rotation\":",sink);array(d->xmat+9*b,9);
  /* XBODY is the body origin, BODY would return inertial-COM velocity.
   * World-aligned, angular first, then linear. Read-only public API. */
  mjtNum v[6]={0};velocity_real(m,d,mjOBJ_XBODY,b,v,0);
  fputs(",\"origin_velocity_world_angular_linear\":",sink);array(v,6);
  fputc('}',sink);
}
static void json_string(FILE* file,const unsigned char* value) {
  if(!value){fputs("null",file);return;}
  fputc('"',file);
  for(const unsigned char* p=value;*p;p++) {
    if(*p=='"'||*p=='\\')fputc('\\',file);
    if(*p<32)fprintf(file,"\\u%04x",*p);else fputc(*p,file);
  }
  fputc('"',file);
}
void mjr_makeContext(const mjModel* m,mjrContext* context,int fontscale) {
  /* The genuine render thread owns a current GL context here. Observe only:
   * no extra rendering/stepping and no model, control or GL-state writes. */
  void (*real_context)(const mjModel*,mjrContext*,int)=dlsym(RTLD_NEXT,"mjr_makeContext");
  if(!real_context)_exit(125);
  real_context(m,context,fontscale);
  const char* path=getenv("ROBOTSIM_NATIVE_GRAPHICS_OUT");
  if(!path||!*path)return;
  const unsigned char* (*get_string)(unsigned int)=dlsym(RTLD_DEFAULT,"glGetString");
  if(!get_string) {
    typedef void (*GenericGlProc)(void);
    GenericGlProc (*get_proc)(const char*)=dlsym(RTLD_DEFAULT,"glfwGetProcAddress");
    if(get_proc)get_string=(const unsigned char* (*)(unsigned int))get_proc("glGetString");
  }
  FILE* file=fopen(path,"wx");if(!file)return;
  fprintf(file,"{\"classification\":\"READ_ONLY_ACTUAL_NATIVE_GL_CONTEXT\",\"pid\":%ld,\"vendor\":",(long)getpid());
  json_string(file,get_string?get_string(0x1F00):NULL);fputs(",\"renderer\":",file);
  json_string(file,get_string?get_string(0x1F01):NULL);fputs(",\"version\":",file);
  json_string(file,get_string?get_string(0x1F02):NULL);fputs("}\n",file);fclose(file);
}
void mj_step(const mjModel* m,mjData* d) {
  pthread_once(&once,init);
  long long begin=mono_ns();
  if(last_step_end_ns && begin-last_step_end_ns>max_call_gap_ns)max_call_gap_ns=begin-last_step_end_ns;
  step_real(m,d);
  long long physics_end=mono_ns();
  if(physics_end-begin>max_step_duration_ns)max_step_duration_ns=physics_end-begin;
  last_step_end_ns=physics_end;
  if (!sink || count>=12000) return;
  if (d->time<previous) {generation++;next_time=-1;memset(peak_qvel,0,sizeof(peak_qvel));qvel_window_start=d->time;}
  if(m->njnt>4096){fprintf(stderr,"NATIVE_AUDIT_JOINT_WINDOW_CAPACITY\n");return;}
  for(int j=0;j<m->njnt;j++) {
    if(m->jnt_type[j]==mjJNT_FREE || m->jnt_type[j]==mjJNT_BALL)continue;
    mjtNum v=d->qvel[m->jnt_dofadr[j]];if(v<0)v=-v;
    if(v>peak_qvel[j])peak_qvel[j]=v;
  }
  previous=d->time;
  if (d->time+1e-10<next_time) return;
  next_time=d->time+.02;count++;
  char* buffer=NULL;size_t length=0;FILE* logfile=sink;
  sink=open_memstream(&buffer,&length);
  if(!sink){sink=logfile;return;}
  long long started=mono_ns();
  fprintf(sink,"{\"host_monotonic_ns\":%lld,\"process_pid\":%ld,\"sim_time_s\":%.17g,\"generation\":%d,\"engine_version\":%d,\"timestep_s\":%.17g,\"model_keyframe_count\":%d,",started,(long)getpid(),(double)d->time,generation,version_real(),(double)m->opt.timestep,m->nkey);
  fprintf(sink,"\"solver_options\":{\"noslip_iterations\":%d,\"cone\":%d,\"impratio\":%.17g,\"iterations\":%d},\"bodies\":{",m->opt.noslip_iterations,m->opt.cone,(double)m->opt.impratio,m->opt.iterations);
  body(m,d,"pelvis");fputc(',',sink);body(m,d,"R_omnipicker_base_link");fputc(',',sink);body(m,d,"test_cylinder");fputc(',',sink);body(m,d,"head_pitch_link");
  fputs("},\"joints\":{",sink);
  int emitted_joint=0;
  for (int j=0;j<m->njnt;j++) {
    const char* joint_name=name_real(m,mjOBJ_JOINT,j);
    if(!joint_name || m->jnt_type[j]==mjJNT_FREE || m->jnt_type[j]==mjJNT_BALL)continue;
    fprintf(sink,"%s\"%s\":{\"q\":%.17g,\"qvel\":%.17g,\"bias_force\":%.17g,\"max_abs_qvel_since_previous_sample\":%.17g}",emitted_joint++?",":"",joint_name,(double)d->qpos[m->jnt_qposadr[j]],(double)d->qvel[m->jnt_dofadr[j]],(double)d->qfrc_bias[m->jnt_dofadr[j]],(double)peak_qvel[j]);
  }
  memset(peak_qvel,0,sizeof(peak_qvel));
  fprintf(sink,"},\"joint_velocity_window_start_sim_time_s\":%.17g,\"claw_actuators\":{",qvel_window_start);
  qvel_window_start=d->time;
  const char* acts[]={"motor_L_claw_joint","motor_R_claw_joint"};
  for(int i=0;i<2;i++) {
    int a=id_real(m,mjOBJ_ACTUATOR,acts[i]);fprintf(sink,"%s\"%s\":",i?",":"",acts[i]);
    if(a<0)fputs("null",sink);
    else fprintf(sink,"{\"ctrl\":%.17g,\"force\":%.17g}",(double)d->ctrl[a],(double)d->actuator_force[a]);
  }
  fprintf(sink,"},\"active_contact_count\":%d,\"contacts\":[",d->ncon);
  int bottle=id_real(m,mjOBJ_BODY,"test_cylinder");
  int table=id_real(m,mjOBJ_BODY,"raicom_table_table"),emitted=0;
  if(table<0)table=id_real(m,mjOBJ_BODY,"raicom_table");
  for(int i=0;i<d->ncon;i++) {
    const mjContact* c=d->contact+i;
    if(c->geom1<0 || c->geom2<0) continue;
    int b1=m->geom_bodyid[c->geom1],b2=m->geom_bodyid[c->geom2];
    if(b1!=bottle && b2!=bottle && b1!=table && b2!=table) continue;
    const char* n1=name_real(m,mjOBJ_BODY,b1);const char* n2=name_real(m,mjOBJ_BODY,b2);
    const char* g1=name_real(m,mjOBJ_GEOM,c->geom1);const char* g2=name_real(m,mjOBJ_GEOM,c->geom2);
    /* Never let anonymous strings produce malformed JSON. Source identifiers
     * used here are registry-controlled names, not arbitrary user prose. */
    mjtNum f[6]={0};force_real(m,d,i,f);
    fprintf(sink,"%s{\"body1\":\"%s\",\"body2\":\"%s\",\"geom1\":\"%s\",\"geom2\":\"%s\",\"distance_m\":%.17g,\"efc_address\":%d,\"position_m\":",emitted++?",":"",n1?n1:"",n2?n2:"",g1?g1:"",g2?g2:"",(double)c->dist,c->efc_address);
    array(c->pos,3);fputs(",\"contact_frame\":",sink);array(c->frame,9);
    fputs(",\"local_force_torque\":",sink);array(f,6);
    fputs(",\"contact_friction\":",sink);array(c->friction,5);
    mjtNum v1[6]={0},v2[6]={0},rel[3]={0},local[3]={0};
    velocity_real(m,d,mjOBJ_BODY,b1,v1,0);velocity_real(m,d,mjOBJ_BODY,b2,v2,0);
    for(int k=0;k<3;k++) {
      int j=(k+1)%3,l=(k+2)%3;
      rel[k]=v2[k+3]-v1[k+3]+v2[j]*(c->pos[l]-d->xipos[3*b2+l])-v2[l]*(c->pos[j]-d->xipos[3*b2+j])-v1[j]*(c->pos[l]-d->xipos[3*b1+l])+v1[l]*(c->pos[j]-d->xipos[3*b1+j]);
    }
    for(int k=0;k<3;k++) for(int j=0;j<3;j++) local[k]+=c->frame[3*k+j]*rel[j];
    fputs(",\"relative_velocity_contact_m_s\":",sink);array(local,3);
    fprintf(sink,",\"contact_dimension\":%d",c->dim);
    double support=0;
    if(b1==bottle || b2==bottle) for(int k=0;k<3;k++) support+=(b2==bottle?1:-1)*c->frame[k*3+2]*f[k];
    fprintf(sink,",\"signed_bottle_world_z_force_n\":%.17g}",support);
  }
  fprintf(sink,"],\"observer_work_ns\":%lld,\"timing\":{\"max_between_step_calls_ns\":%lld,\"max_physics_step_ns\":%lld,\"previous_observer_io_ns\":%lld,\"max_observer_io_ns\":%lld}}\n",mono_ns()-started,max_call_gap_ns,max_step_duration_ns,last_io_ns,max_io_ns);
  max_call_gap_ns=0;max_step_duration_ns=0;
  fclose(sink);sink=logfile;
  long long io_started=mono_ns();
  fwrite(buffer,1,length,sink);fflush(sink);
  if(*latest_path){
    FILE* latest=fopen(latest_tmp,"w");
    if(latest){fwrite(buffer,1,length,latest);fclose(latest);rename(latest_tmp,latest_path);}
  }
  free(buffer);
  last_io_ns=mono_ns()-io_started;if(last_io_ns>max_io_ns)max_io_ns=last_io_ns;
  last_step_end_ns=mono_ns();
}
