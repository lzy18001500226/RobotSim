std_msgs/Header header
string schema_version
string model_identity
string generation
uint64 sequence
string source_id

uint8 COMMAND_TORQUE=1
uint8 command_type
string[] joint_names
float64[] effort_nm
