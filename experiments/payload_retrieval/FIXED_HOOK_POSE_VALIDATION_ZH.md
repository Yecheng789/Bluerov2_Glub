# 真实水池固定勾取位姿验证

当前唯一目标是完成导师在 `0:20:00` 提出的最低可交付实验：人工遥控
BlueROV 勾住目标，记录此时稳定的原始 MoCap 位姿，再自动先到达目标前方同深度的
预接近点、对准目标姿态，沿低速直线前进到记录位姿，连续保持记录姿态 5 秒，再沿
原直线后退到预接近点，保存数据并录制清晰视频。本阶段不包含 YOLO、在线目标估计、
自动穿钩、返航或靠泊。

已确认：MoCap 刚体名是 `glub_fb`，原始位姿话题是 `/mocap/glub_fb/pose`；
PX4 namespace 仍是 `/glub`，池体尺寸是 `9 x 5 x 3 m`。这两个名字属于不同系统，
`/glub/fmu/...` 中的 `glub` 不随 MoCap 刚体名变化。当前固定目标实验默认采用
`NED/FRD`、`standard`、MoCap 池界 `0 9 -2.5 2.5 0 3`、MAVLink 目标 ID
`3/1` 和目标文件 `hooked_box_target_pose_20260802_195146.json`。本工作区的
`src/px4_msgs` 已从当前 `/home/yecheng/PX4-Autopilot/msg` 同步：ACK v0、Status v1；
real launch 会在版本不匹配时直接拒绝启动。

## 0. 下水前必须核对的信息

本实验的默认值已经写入启动文件：

- `mocap_world_frame=ned`；
- `mocap_body_frame=frd`；
- `robot_type=standard`；
- `pool_bounds_mocap='0 9 -2.5 2.5 0 3'`；
- `pre_approach_distance_m=0.50`；
- 进入最终直线前要求独立深度误差
  `fixed_hook_depth_tolerance_m=0.03`；
- 到预接近点 `pre_approach_speed_mps=0.06`，最终直线接近
  `final_approach_speed_mps=0.04`，姿态参考上限
  `traj_angular_speed_deg_s=8.0`；
- `final_pose_hold_s=5.0`、`retreat_speed_mps=0.04`；
- `thrust_sat=0.12`、`torque_sat=0.02`；
- 稳态位置误差补偿为 `3.0 N/(m*s)`，积分力偏置每轴最多为该轴
  `0.07`，且 MPC 与积分之和仍受 `thrust_sat` 限制；
- 位置/姿态使用 MoCap，角速度使用
  `/glub/fmu/out/vehicle_odometry.angular_velocity`（BODY_FRD）；
- `max_mocap_coast_sec=2.0`；
- `target_system_id=3`、`target_component_id=1`。

任何现场坐标系、配重、池体原点或飞控 ID 改动后，都必须覆盖对应 launch 参数并重新
做静态检查；不要继续沿用这些默认值。

还必须从真实飞控核对 MAVLink `system_id/component_id`，以及水下适用的
`COM_OF_LOSS_T`、`COM_OBL_RC_ACT`。这些值不能由 ROS namespace 或池尺寸推导。

## 1. 终端 0：编译一次

```bash
cd /home/yecheng/bluerov_ws
source /opt/ros/jazzy/setup.bash

export ACADOS_SOURCE_DIR=/home/yecheng/acados
export LD_LIBRARY_PATH=/home/yecheng/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

colcon build --packages-select px4_msgs --symlink-install --cmake-clean-cache
colcon build --packages-select bluerov2_control --symlink-install

source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
```

每次重新编译后，所有新终端都要重新执行：

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
```

## 2. 终端 1：DDS Agent（仅在机器人使用本机 UDP 8888 时）

先检查，避免重复启动两个 Agent：

```bash
pgrep -af 'micro-xrce-dds-agent|MicroXRCEAgent'
ss -lunp | grep -E ':8888'
```

若没有进程，并且真实 PX4 确认配置为连接这台电脑的 UDP 8888，再运行：

```bash
micro-xrce-dds-agent udp4 -p 8888
```

如果 PX4 使用串口、另一台电脑或不同端口，不要照抄这一行，必须使用现场配置。

## 3. 终端 2：先定位并验证 PX4 DDS 输出和消息版本

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic list | sort | grep -E '^/glub/fmu/(in|out)/'
ros2 topic list | sort | grep -E \
  '^/glub/fmu/out/(vehicle_command_ack|vehicle_control_mode|vehicle_status)(_v[0-9]+)?$'

ros2 topic info -v /glub/fmu/out/vehicle_control_mode
ros2 topic echo /glub/fmu/out/vehicle_control_mode \
  --once --qos-profile sensor_data
```

2026-07-21 早先曾观察到 `/glub/fmu/in/*` 有 PX4 subscription、所有关键 output 不存在；
稍后主机 ROS 图又出现了 `vehicle_control_mode`、无后缀 `vehicle_command_ack` 和
`vehicle_status_v1`，但 `vehicle_control_mode --once` 仍未收到数据。因此 DDS 端点“出现在
topic list”还不等于 PX4 正在持续发送有效反馈。

本工作区的 `src/px4_msgs` 已按 PX4 官方手工同步方法，从当前
`/home/yecheng/PX4-Autopilot/msg` 复制，当前应为 ACK v0、Status v1。不要再从
`/home/yecheng/px4_ws` 覆盖它；该工作区仍是 ACK v1、Status v2。每个现实实验终端都必须
使用 `source_fixed_hook_real.bash`，脚本会清理旧 overlay 并校验实际加载的是 ACK/Status
`0/1`。

在 QGC 的 MAVLink Console 中执行以下只读命令，并保存完整输出：

```text
ver all
uxrce_dds_client status
listener vehicle_control_mode 3
listener vehicle_status 1
param show UXRCE_DDS_CFG
param show UXRCE_DDS_AG_IP
param show UXRCE_DDS_PRT
param show UXRCE_DDS_DOM_ID
param show COM_OF_LOSS_T
param show COM_OBL_RC_ACT
```

若以后更新或重新烧录飞控，必须重新以 `ver all` 显示的真实固件 checkout 为准，按
`px4_msgs/README.md` 的同步方法复制普通和 `versioned` 消息，再重新编译两个包。不要只
修改 topic 后缀来掩盖消息不匹配。当前版本重新编译命令为：

```bash
cd /home/yecheng/bluerov_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select px4_msgs --symlink-install --cmake-clean-cache
colcon build --packages-select bluerov2_control --symlink-install
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash
```

重新编译并 source 后必须同时满足：

- `vehicle_control_mode` 连续有数据；
- 实际 ACK 和 Status 话题的 `_vN` 后缀与已编译消息版本一致；
- `vehicle_status` 能正确读出 `system_id/component_id`；
- `VehicleControlMode` 的 `flag_armed`、`flag_control_offboard_enabled` 能随 QGC 状态变化。

版本对齐后用下面的命令读取 MAV IDs，并把结果用于第 8 节的 launch 参数：

```bash
STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")
ros2 topic info -v "$STATUS_TOPIC"
ros2 topic echo "$STATUS_TOPIC" --once --qos-profile sensor_data
```

这些条件全部满足前，不要启动现实 MPC 实验。QGC GeoClue 警告不是这个 DDS 故障的
原因。

## 4. 终端 3：启动 QGroundControl

```bash
cd /home/yecheng/Downloads
./QGroundControl-x86_64.AppImage
```

确认 QGC 连接的是与 ROS 2 `vehicle_status` 相同 `system_id/component_id` 的真实飞控；
QGC 本身不显示 ROS namespace。QGC 的 GeoClue 警告不会解释 ROS 2
输出话题消失。`Unknown 'function': "center"` 和 joystick axis 警告也不是 DDS 消失的
直接原因，但本实验依赖人工勾取和紧急接管，因此下水前必须单独实测 joystick 六自由度、
模式切换、DISARM/kill 都按预期工作；没有通过时不能开始实验。

## 5. 终端 4：检查原始 MoCap

```bash
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic info -v /mocap/glub_fb/pose
ros2 topic echo /mocap/glub_fb/pose --once --qos-profile sensor_data
ros2 topic hz /mocap/glub_fb/pose
```

`topic hz` 会持续运行；观察至少 20 秒后按 `Ctrl-C`。必须确认：

- publisher 数量是 1；
- 位姿连续更新，header stamp 非零且递增；
- `frame_id` 非空且不变化；
- 机器人静止时没有 NaN、零四元数或明显跳变；
- publisher 是 `RELIABLE` 或 `BEST_EFFORT` 均可，新订阅端兼容两者。

### 实验室电脑出现 `sequence size exceeds remaining buffer` 时

这不是终端显示缓冲区满，而是 Fast-CDR 无法按本地接口定义反序列化收到的 DDS 数据。
若 `robot_state_publisher`、`mocap_qualisys_node` 和 `watertank_tf_utils` 等不同进程同时刷
这句话，应立即停止有推力实验。当前控制电脑是 ROS 2 Jazzy；实验室安装说明使用
ROS 2 Humble，而 ROS 官方不支持依赖跨发行版 DDS 通信。两台电脑分别检查：

```bash
echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-default}"
python3 -c "import rclpy; print(rclpy.get_rmw_implementation_identifier())"
```

现实实验中两端必须使用同一 ROS 发行版、同一 RMW，并且只 source 这一套环境。推荐把
Qualisys 发布栈也编译/运行在 Jazzy。若实验室栈只能使用 Humble，则必须先分离 DDS
Domain，再建立经过验证、只转发 `/mocap/glub_fb/pose` 的桥；仅修改 `ROS_DOMAIN_ID`
却不建立桥会让 MPC 完全收不到 MoCap。恢复后先执行：

```bash
ros2 daemon stop
ros2 topic info -v /mocap/glub_fb/pose
ros2 topic echo /mocap/glub_fb/pose --once --qos-profile sensor_data
timeout 20s ros2 topic hz /mocap/glub_fb/pose
```

确认不再出现反序列化错误、publisher 只有一个且频率稳定后，才能继续。

再连续观察至少 5 秒，而不是只看一帧；同时比较 MoCap stamp 与本机系统时间，确认两者
属于同一时钟域，延迟小于 `0.20 s`：

```bash
date +%s.%N
ros2 topic echo /mocap/glub_fb/pose --qos-profile sensor_data
```

观察 `header.stamp`、`frame_id` 和 pose 后按 `Ctrl-C`。如果两台机器未同步，记录器和正式
EKF 会拒绝旧时间戳；不要通过关闭时间检查掩盖时钟问题。

如果机器人水平静止时姿态仍明显错误，可先运行：

```bash
ros2 run bluerov2_control calibrate_mocap_orientation_correction \
  --topic /mocap/glub_fb/pose \
  --samples 160
```

保存输出的 `orientation_correction_quat_xyzw`。同一个目标文件和正式实验必须使用同一
组刚体定义、marker 安装和姿态修正。

## 6. 终端 5：人工勾住后记录目标

先用人工遥控勾住箱体，让机器人和箱体稳定静止；此时不要运行自动 MPC。每次正式实验
新建一个文件，不覆盖旧目标。开始人工操作前先确认以下四个控制输入的
`Publisher count` 都是 0；其中 PX4 的 subscription 不算冲突：

```bash
ros2 topic info -v /glub/fmu/in/offboard_control_mode
ros2 topic info -v /glub/fmu/in/vehicle_command
ros2 topic info -v /glub/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /glub/fmu/in/vehicle_torque_setpoint
```

然后记录：

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

TARGET_CFG=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_$(date +%Y%m%d_%H%M%S).json

ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub_fb/pose \
  --message-type pose \
  --samples 160 \
  --timeout-sec 20 \
  --max-message-age-sec 0.20 \
  --output-file "$TARGET_CFG"

python3 -m json.tool "$TARGET_CFG"
```

只有采样时间、时间戳、位置标准差和姿态离散度全部通过时才会写文件。记下最后打印的
绝对路径；换终端后需要重新设置 `TARGET_CFG`。从这一刻开始箱体和目标把手不得移动，
否则目标文件立即作废，必须重新勾住并记录。

刚体改名后，旧目标 JSON 中记录的 `/mocap/glub/...` 或 `/mocap/glub_4/...`
是历史来源证据，不能直接把字符串改成 `glub_fb` 后继续使用。正式 launch 会拒绝
这些旧目标；必须按本节从 `/mocap/glub_fb/pose` 重新采集。

## 7. 终端 6：先录 rosbag

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

BAG_DIR=/home/yecheng/bluerov_ws/fixed_hook_bags/fixed_hook_$(date +%Y%m%d_%H%M%S)
mkdir -p /home/yecheng/bluerov_ws/fixed_hook_bags

ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
STATUS_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleStatus as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_status'+(f'_v{v}' if v else ''))")
ros2 topic info -v "$ACK_TOPIC"
ros2 topic info -v "$STATUS_TOPIC"

ros2 bag record -o "$BAG_DIR" --topics \
  /mocap/glub_fb/pose \
  /mocap/glub_fb/odom_ekf_fixed_hook \
  /mocap/glub_fb/vehicle_odometry_fixed_hook \
  /glub/fmu/in/offboard_control_mode \
  /glub/fmu/in/vehicle_command \
  /glub/fmu/in/vehicle_thrust_setpoint \
  /glub/fmu/in/vehicle_torque_setpoint \
  "$ACK_TOPIC" \
  /glub/fmu/out/vehicle_control_mode \
  /glub/fmu/out/vehicle_odometry \
  "$STATUS_TOPIC" \
  /glub/fmu/out/failsafe_flags \
  /bluerov2/fixed_hook/offboard_request_enable \
  /bluerov2/fixed_hook/mission_enable \
  /bluerov2/fixed_hook/controller_heartbeat \
  /bluerov2/trial_event \
  /parameter_events
```

开始录包后，在箱体保持不动的前提下人工脱钩，把 ROV 移到池内安全起点，最好靠近
launch 计算出的预接近点，然后在 QGC 中 DISARM。确认 ROV、箱体和缆绳都静止后才
启动下一节。现在不再要求起点到最终目标必须小于 `0.75 m`，但起点仍必须位于下述
收缩后的池体工作边界内，且到预接近点的直线路径必须无障碍、缆绳无缠绕。

## 8. 终端 7：启动固定目标验证（仍不会运动）

当前实验参数已固化为 launch 默认值：

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 launch bluerov2_control fixed_hook_pose_validation.launch.py
```

若重新标定后得到非空姿态修正，只需增加这个覆盖参数：

```bash
ros2 launch bluerov2_control fixed_hook_pose_validation.launch.py \
  orientation_correction_quat_xyzw:='x y z w'
```

`standard` 模型约有 `9.82 N` 正浮力；当前 `thrust_sat=0.12` 给深度控制保留了
更多余量，但并不证明真实配重安全。必须先确认真实质量/排水体积，并做有安全绳的
低风险垂向保持测试。代码会拒绝 `standard + thrust_sat<0.075`。

启动文件会校验目标 JSON，将目标和实时位姿用同一个转换映射到 NED/FRD，启动 EKF、
适配器、真实 Fossen MPC、日志器和 Offboard 管理器。它会把六个原始 MoCap 池界转换成
NED，并将每一面收缩 `0.25 m`；目标或实时 ROV 中心越过有效边界时会拒绝/停止任务。
它还会按最终目标 yaw 的水平机体前向量计算预接近点：
`pre = goal - 0.50 * [cos(yaw), sin(yaw), 0]`，并强制 `pre_z=goal_z`。当前目标的默认
预接近点约为 NED `[4.845, 0.078, 1.728] m`；预接近点若落在工作边界外，launch 会
直接拒绝启动。

任务采用连续位姿参考流程：先移动到预接近点并对准记录姿态，再沿目标机体前向进入
最终记录位姿；只有三维位置误差不超过 `0.05 m`、独立深度误差不超过 `0.03 m`，
且姿态误差不超过 `5 deg`，才会从预接近点开始向前；最终点和后退终点也使用同一
深度门槛。达到最终位姿后才开始连续
5 秒保持，保持期间漂出容差会重新计时。保持完成后，机器人保持同一记录姿态并沿原
直线后退到预接近点，随后在那里定点保持。移动到预接近点的速度默认
`0.06 m/s`，最后 `0.50 m` 直线进入和后退速度均为 `0.04 m/s`，姿态参考变化速度
上限默认 `8 deg/s`。分阶段速度避免为了缩短前段路程而让靠近箱体的阶段也加速。

最终前进参考严格使用配置中的 `pre-hook -> recorded hook` 两个端点，后退参考使用完全
相反的两个端点；所有中间参考点的 NED Z 都固定为记录 Hook 的 Z，姿态也固定为记录
姿态。代码不会再把阶段切换瞬间最多数厘米的实测误差当作新轨迹起点，因此不会生成
斜向进入或另一条返回线。预接近点已经学到的 world-Z 浮力/缆绳补偿会在水平前进和
后退期间冻结并继续施加，但不会继续积分；X/Y 积分在换阶段时清零。任务关闭、odom
失效或其他 fail-closed 路径仍会把全部积分清零。

有限时域 MPC 本身没有外部扰动力状态；缆绳拉力、浮力或模型误差可能使机器人在
参考轨迹结束后停在目标附近但仍有明显位置误差。现实 launch 因此只在名义轨迹已经
结束、目标误差不超过 `0.50 m` 时启用有界位置积分。默认增益为 `3.0 N/(m*s)`，积分
偏置最多占每个物理力轴的 `0.07`；积分偏置与 MPC 原始力相加后再次按
`thrust_sat=0.12` 限幅。固定 Hook 的水平前进/后退会按上段所述只继承冻结的 Z
补偿；启动其他新轨迹、任务关闭或任何 fail-closed 重置仍会清零全部积分。位置误差
超过 `0.50 m` 时不会积分，以免用持续增大的力掩盖坐标系错误、机械阻挡或缆绳缠绕。

MPC 的位置、姿态和线速度仍来自 MoCap；用于姿态阻尼的角速度改为 PX4
`/glub/fmu/out/vehicle_odometry.angular_velocity` 的 BODY_FRD 数据。该数据超过
`0.10 s` 未更新时，适配器会停止发布 MPC odometry，使控制链 fail-closed。不要把
PX4 FRD 角速度直接接入旧的 MAVROS FLU IMU 路径，否则 Y/Z 会被错误翻转。
此时：

- `auto_arm=false`；
- Offboard 请求许可仍是 false；
- MPC 运动许可仍是 false；
- 控制器只发送零 thrust/torque；
- 任一关键进程退出会关闭整个 launch；
- MPC executor/ACADOS 卡住时，独立进程会停止 PX4 Offboard heartbeat。

旧名字 `fixed_hook_mpc_june23.launch.py` 现在只是这个安全流程的兼容包装，不再加载
June 23 的历史目标。

旧版 `0.75 m` 和 `45 deg` 是一次性的启动合理性检查，用来在坐标系、目标文件或姿态
方向配错时阻止长距离/大角度误动作。它们只比较“起点到当时活动目标”的球形距离和
全姿态误差，无法描述“先到目标前方、再直线进入”的几何流程。现在 real launch 默认
把这两个旧门限设为 `0`（在 MPC 中表示关闭），改由已验证的预接近点、分阶段切换、
池体工作边界、MoCap 跳变检查和控制输出限幅共同约束。需要额外保守检查时仍可在启动
命令中覆盖为非零值，例如 `max_initial_goal_distance_m:=1.0`；不要通过调大工作边界或
关闭 MoCap/输出安全检查来代替正确摆放机器人。

### 本实验必须独占控制话题

新启动文件运行时，绝对不要同时运行以下程序：

- `stabilized_control_real.launch.py`；
- 单独的 `offboard_enable_real.launch.py`；
- `fixed_hook_mpc_june23.launch.py`（它与新名字二选一）；
- 任何旧 PID、旧 MPC 或 remap 到 `/glub` 的 controller；
- 任何单独的 `offboard_heartbeat_wrench`、`offboard_heartbeat_attitude` 或
  `offboard_heartbeat_actuator`。

ROS 2 不会在多个控制 publisher 之间仲裁。上述程序会与本流程争用 heartbeat、模式命令、
thrust 或 torque。QGC、rosbag 和纯订阅监视器可以并存；`real_wrench_watchdog` 也只适合
观察，它不会在失流时清零、退出 Offboard 或替代硬件急停。

## 9. 终端 8：启动后的静态检查

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 node list | sort | grep -E 'fixed_hook|mocap_ekf|offboard_enable'

ros2 topic info -v /glub/fmu/in/offboard_control_mode
ros2 topic info -v /glub/fmu/in/vehicle_command
ros2 topic info -v /glub/fmu/in/vehicle_thrust_setpoint
ros2 topic info -v /glub/fmu/in/vehicle_torque_setpoint

ros2 topic hz /mocap/glub_fb/odom_ekf_fixed_hook
ros2 topic hz /glub/fmu/out/vehicle_odometry
ros2 topic echo /glub/fmu/out/vehicle_odometry \
  --once --qos-profile sensor_data
```

观察频率后按 `Ctrl-C`，再检查 MPC 参数：

```bash
ros2 param get /mpc_fixed_hook_pose_validation model_type
ros2 param get /mpc_fixed_hook_pose_validation robot_type
ros2 param get /mpc_fixed_hook_pose_validation goal_x
ros2 param get /mpc_fixed_hook_pose_validation goal_y
ros2 param get /mpc_fixed_hook_pose_validation goal_z
ros2 param get /mpc_fixed_hook_pose_validation use_pre_approach_waypoint
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_x
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_y
ros2 param get /mpc_fixed_hook_pose_validation pre_approach_z
ros2 param get /mpc_fixed_hook_pose_validation traj_angular_speed_rad_s
ros2 param get /mpc_fixed_hook_pose_validation final_pose_hold_s
ros2 param get /mpc_fixed_hook_pose_validation return_to_pre_approach_after_hold
ros2 param get /mpc_fixed_hook_pose_validation backward_pass_speed_mps
ros2 param get /mpc_fixed_hook_pose_validation torque_sat
ros2 param get /mpc_fixed_hook_pose_validation require_mission_enable
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_enable
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_xmin_m
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_xmax_m
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_ymin_m
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_ymax_m
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_zmin_m
ros2 param get /mpc_fixed_hook_pose_validation operating_bounds_zmax_m
```

四个 PX4 input topic 都应只有本流程对应的一个 publisher。若同时存在旧 heartbeat、PID、
stabilized controller 或第二个 MPC，立即全部停止，不能继续。

还要在 QGC 中确认现场采用的 `COM_OF_LOSS_T` 和 `COM_OBL_RC_ACT`：当本流程因 MoCap、
MPC 或 DDS 故障撤销 heartbeat 后，真实飞控必须进入导师和现场安全员认可、且水下确实
可执行的动作。默认的 Position/Return 等飞行器策略未必适合没有 PX4 本地位置输入的
水下机器人，不要在这里盲目写参数。

## 10. 终端 9：显式请求 Offboard

先清除可能由启动期间瞬时掉线产生的安全锁，并启动 ACK 监视窗口：

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic pub --once \
  /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: false}'

ros2 topic pub --once \
  /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: false}'

ACK_TOPIC=$(python3 -c "from px4_msgs.msg import VehicleCommandAck as M; v=int(M.MESSAGE_VERSION); print('/glub/fmu/out/vehicle_command_ack'+(f'_v{v}' if v else ''))")
ros2 topic info -v "$ACK_TOPIC"
ros2 topic echo "$ACK_TOPIC" \
  --qos-profile sensor_data
```

完成前述检查、确保推进器区域无人，并准备好 QGC 模式切换和硬件急停后，在另一个已
source 的终端运行：

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 topic pub --once \
  /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: true}'
```

必须看到匹配的 `VEHICLE_CMD_DO_SET_MODE` ACK，并在 QGC 和
`/glub/fmu/out/vehicle_control_mode` 中都确认 Offboard。管理器会继续重试，直到实际模式
反馈确认；它永远不会自动 ARM。

在另一个终端读取真实反馈；若没有消息，或 `flag_control_offboard_enabled` 不是 `true`，
不要继续：

```bash
ros2 topic echo /glub/fmu/out/vehicle_control_mode \
  --once --qos-profile sensor_data
```

## 11. QGC 人工 ARM，然后终端 10 才允许 MPC 运动

先在 QGC 人工 ARM，然后再次读取真实反馈：

```bash
ros2 topic echo /glub/fmu/out/vehicle_control_mode \
  --once --qos-profile sensor_data
```

只有 `flag_armed: true` 和 `flag_control_offboard_enabled: true` 同时成立，才能继续。
确认 ROV、最终目标和 launch 打印的预接近点都在收缩后的工作边界内，起点到预接近点
的直线路径无障碍、缆绳无缠绕，并且坐标方向的小幅测试正确，再标记事件并启动轨迹：

```bash
cd /home/yecheng/bluerov_ws
source /home/yecheng/bluerov_ws/src/bluerov2_control/scripts/source_fixed_hook_real.bash

ros2 run bluerov2_control mark_trial_event start \
  --note 'fixed hook pose validation enabled'

ros2 topic pub --once \
  /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: true}'
```

控制器会先到默认距最终目标 `0.50 m`、与目标同深度的预接近点并完成姿态对准，再自动
切换到最终直线进入阶段。到达记录位姿后连续保持 5 秒；如果保持期间漂出位置/姿态
容差，计时会清零。之后控制器保持记录姿态，沿原路径直线后退到预接近点并在那里
定点保持。观察到完成日志后：

```bash
ros2 run bluerov2_control mark_trial_event success \
  --note 'target pose reached and held stably'
```

## 12. 正常停止和紧急停止

正常停止时先撤销运动许可：

```bash
ros2 topic pub --once \
  /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: false}'

ros2 topic pub --once \
  /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: false}'

ros2 run bluerov2_control mark_trial_event stop \
  --note 'operator stopped validation'
```

然后在 QGC 中 DISARM 并切离 Offboard，最后依次对 launch 和 rosbag 按 `Ctrl-C`。
`offboard_request_enable=false` 只会停止后续模式请求，不会替你切换一个已经生效的 PX4
模式。

紧急情况优先使用 QGC/硬件 kill 或 DISARM，同时停止 launch。ROS 2 的 false 命令是
附加保护，不应代替硬件急停。

若日志出现 `Mission enable revoked`、`controller-health latch`、MoCap/odom stale、姿态
跳变或 ACADOS failure，不允许等待它自动恢复。先在 QGC DISARM，再依次发送：

```bash
ros2 topic pub --once \
  /bluerov2/fixed_hook/mission_enable \
  std_msgs/msg/Bool '{data: false}'

ros2 topic pub --once \
  /bluerov2/fixed_hook/offboard_request_enable \
  std_msgs/msg/Bool '{data: false}'
```

查明并修复原因后，必须重新执行静态检查，再按“Offboard request true -> QGC ARM ->
mission true”的顺序开始；代码不会在健康链路恢复后自行重新许可运动。

## 13. 视频和数据验收

视频至少连续展示：人工勾住并记录、机器人移动到安全起点、QGC 的 Armed/Offboard
状态、第一阶段到达并对准预接近点、第二阶段沿直线进入、最终稳定保持以及安全停止。
避免只拍局部镜头而看不到箱体、ROV 和接近方向。

每次 launch 会在以下目录新建带微秒时间戳的 trial：

```bash
ls -lt /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials | head
```

离线分析一个明确的 trial。这里必须使用转换后的 NED odom，并填写 launch 打印/记录的
NED 目标和 NED 安全边界；不能把原始 MoCap 数字直接混进来：

```bash
export GOAL_NED='REPLACE_WITH_x,y,z'
export GOAL_QUATERNION_WXYZ='REPLACE_WITH_qw,qx,qy,qz'
export POOL_BOUNDS_NED='REPLACE_WITH_xmin,xmax,ymin,ymax,zmin,zmax'

ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS_ffffff \
  --pose-source odom \
  --payload-target="$GOAL_NED" \
  --target-quaternion-wxyz="$GOAL_QUATERNION_WXYZ" \
  --hold-window-s=5.0 \
  --tank-bounds="$POOL_BOUNDS_NED"
```

`GOAL_QUATERNION_WXYZ` 必须填写 launch 使用并记录在 trial metadata `notes` 中的
`goal_quaternion_wxyz_ned_frd`，顺序是 `qw,qx,qy,qz`；不能填写原始 MoCap 的
`x,y,z,w`。分析会报告初始、最小、最终全姿态角误差，以及最后 5 秒的 RMS 和最大
姿态误差，并生成 `orientation_error.png`（未使用 `--no-plots` 时）。
