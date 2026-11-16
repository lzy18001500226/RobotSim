/* SIMULATION_ONLY installed MC native McBodyPose publisher. No velocity topic. */
#define _POSIX_C_SOURCE 200809L
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rosidl_runtime_c/string_functions.h>
#include <rosidl_typesupport_introspection_c/message_introspection.h>
#include <rosidl_typesupport_introspection_c/field_types.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
extern const rosidl_message_type_support_t *rosidl_typesupport_c__get_message_type_support_handle__aimdk_msgs__msg__McBodyPose(void);
extern const rosidl_message_type_support_t *rosidl_typesupport_introspection_c__get_message_type_support_handle__aimdk_msgs__msg__McBodyPose(void);
extern void *aimdk_msgs__msg__McBodyPose__create(void);
extern void aimdk_msgs__msg__McBodyPose__destroy(void *);
typedef rosidl_typesupport_introspection_c__MessageMembers Members;
typedef rosidl_typesupport_introspection_c__MessageMember Member;
static volatile sig_atomic_t stopping=0;
static void stop_signal(int signal_number){(void)signal_number;stopping=1;}
static double monotonic_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const Member *field(const Members *members,const char *name){
 for(uint32_t i=0;i<members->member_count_;i++)if(strcmp(members->members_[i].name_,name)==0)return &members->members_[i];
 return NULL;
}
static const Members *nested(const Member *member){
 return member&&member->members_?(const Members *)member->members_->data:NULL;
}
int main(int argc,char **argv){
 if(argc!=3||strcmp(argv[1],"robotsim_body_height")!=0||access("/.dockerenv",F_OK)||access("/workspace/evidence/native_latest.json",R_OK)){
  fputs("SIMULATION_ONLY body-pose scope/marker rejected\n",stderr);return 2;
 }
 char *tail=NULL;double maximum=strtod(argv[2],&tail);
 if(!tail||*tail||!isfinite(maximum)||maximum<2.||maximum>30.)return 2;
 const rosidl_message_type_support_t *introspection=rosidl_typesupport_introspection_c__get_message_type_support_handle__aimdk_msgs__msg__McBodyPose();
 const Members *members=(const Members *)introspection->data;
 const Member *height=field(members,"stand_height"),*source=field(members,"source"),*header=field(members,"header");
 const Members *hm=nested(header);const Member *stamp=hm?field(hm,"stamp"):NULL;const Members *tm=nested(stamp);
 const Member *sec=tm?field(tm,"sec"):NULL,*ns=tm?field(tm,"nanosec"):NULL;
 if(members->member_count_!=8||!height||height->type_id_!=rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE||height->is_array_||!source||source->type_id_!=rosidl_typesupport_introspection_c__ROS_TYPE_STRING||!sec||!ns){
  fputs("installed McBodyPose ABI mismatch\n",stderr);return 3;
 }
 void *message=aimdk_msgs__msg__McBodyPose__create();if(!message)return 3;
 const char *zero_fields[]={"stand_height","bend_forward","bend_sideways","twist_waist","raise_head","turn_head"};
 for(unsigned int i=0;i<6;i++){
  const Member *f=field(members,zero_fields[i]);
  if(!f||f->type_id_!=rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE||f->is_array_)return 3;
  *(double *)((char *)message+f->offset_)=0.;
 }
 if(!rosidl_runtime_c__String__assign((rosidl_runtime_c__String *)((char *)message+source->offset_),argv[1]))return 3;
 rcl_allocator_t allocator=rcl_get_default_allocator();rcl_init_options_t options=rcl_get_zero_initialized_init_options();
 rcl_context_t context=rcl_get_zero_initialized_context();
 if(rcl_init_options_init(&options,allocator)!=RCL_RET_OK||rcl_init(0,NULL,&options,&context)!=RCL_RET_OK)return 4;
 rcl_node_t node=rcl_get_zero_initialized_node();rcl_node_options_t no=rcl_node_get_default_options();
 if(rcl_node_init(&node,"robotsim_native_body_pose_publisher","/",&context,&no)!=RCL_RET_OK)return 4;
 rcl_publisher_t publisher=rcl_get_zero_initialized_publisher();rcl_publisher_options_t po=rcl_publisher_get_default_options();po.qos.depth=10;
 if(rcl_publisher_init(&publisher,&node,rosidl_typesupport_c__get_message_type_support_handle__aimdk_msgs__msg__McBodyPose(),"/aima/mc/body_pose",&po)!=RCL_RET_OK)return 4;
 signal(SIGTERM,stop_signal);signal(SIGINT,stop_signal);setvbuf(stdin,NULL,_IONBF,0);
 double began=monotonic_s(),last=began,command=0.;unsigned long count=0,nonzero=0,zero=0;int failed=0;
 while(!stopping&&monotonic_s()-began<maximum){
  double now=monotonic_s();struct pollfd input={.fd=STDIN_FILENO,.events=POLLIN};
  while(!stopping&&poll(&input,1,0)>0&&(input.revents&POLLIN)){
   char line[96];if(!fgets(line,sizeof(line),stdin)){stopping=1;command=0.;}
   else{double value=strtod(line,&tail);if(!isfinite(value)||fabs(value)>.100000001||(*tail!='\n'&&*tail!='\0')){failed=1;stopping=1;command=0.;}else{command=value;last=now;}}
  }
  if(now-last>.10)command=0.;
  struct timespec stamp_now;clock_gettime(CLOCK_REALTIME,&stamp_now);
  char *time_base=(char *)message+header->offset_+stamp->offset_;
  *(int32_t *)(time_base+sec->offset_)=(int32_t)stamp_now.tv_sec;*(uint32_t *)(time_base+ns->offset_)=(uint32_t)stamp_now.tv_nsec;
  *(double *)((char *)message+height->offset_)=command;
  if(rcl_publish(&publisher,message,NULL)!=RCL_RET_OK){failed=1;break;}
  count++;if(command!=0.)nonzero++;else zero++;
  struct timespec delay={0,20000000};nanosleep(&delay,NULL);
 }
 double zero_until=monotonic_s()+1.05;
 while(monotonic_s()<zero_until){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);char *base=(char *)message+header->offset_+stamp->offset_;
  *(int32_t *)(base+sec->offset_)=(int32_t)t.tv_sec;*(uint32_t *)(base+ns->offset_)=(uint32_t)t.tv_nsec;
  *(double *)((char *)message+height->offset_)=0.;
  if(rcl_publish(&publisher,message,NULL)==RCL_RET_OK)zero++;
  struct timespec delay={0,20000000};nanosleep(&delay,NULL);
 }
 printf("{\"simulation_only\":true,\"message\":\"aimdk_msgs/msg/McBodyPose\",\"native_message_bytes\":%zu,\"publish_calls\":%lu,\"nonzero_publish_calls\":%lu,\"zero_publish_calls\":%lu,\"mc_receipt_ack\":\"UNAVAILABLE\",\"failed\":%s}\n",members->size_of_,count,nonzero,zero,failed?"true":"false");
 rcl_ret_t status=rcl_publisher_fini(&publisher,&node);(void)status;status=rcl_node_fini(&node);status=rcl_shutdown(&context);status=rcl_context_fini(&context);status=rcl_init_options_fini(&options);
 aimdk_msgs__msg__McBodyPose__destroy(message);return failed?1:0;
}
