std_msgs/Header header
string schema_version
string model_identity
string generation
uint64 sequence

uint8 ORIGIN_GROUND_TRUTH=1
uint8 ORIGIN_MEASUREMENT=2
uint8 ORIGIN_ESTIMATE=3
uint8 origin

uint8 LIFECYCLE_READY=1
uint8 LIFECYCLE_RUNNING=2
uint8 LIFECYCLE_PAUSED=3
uint8 LIFECYCLE_FAILED=4
uint8 lifecycle

uint32 backend_tick
string base_frame
string imu_frame
bool has_root_pose
geometry_msgs/Pose root_pose
bool has_root_twist
geometry_msgs/Twist root_twist
sensor_msgs/JointState joints
sensor_msgs/Imu imu
