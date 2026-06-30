# 水下自主载荷回收实验：数据采集与论文结果分析计划

本目录用于支撑论文题目：

`Controller Design for Autonomous Underwater Payload Retrieval with a Passive Tool and ROVs`

目标是让每次仿真或水池实验都留下可复现的数据包，并能直接生成论文
Results / Analysis 章节需要的指标表、事件时间线和图。

## 论文结果需要采集的数据

每一次完整试验都应覆盖任务链：检测载荷、接近、穿线/挂取、返航到 docking
位置、交付给操作员。建议每个 controller 和每个实验条件重复多次。

必须采集的数据：

- 机器人状态：PX4 odometry 中的位置、姿态四元数、线速度、角速度；水池实验中如果有 Qualisys/mocap，也要记录 mocap odometry。
- 控制输入和输出：`cmd_vel`、thrust setpoint、torque setpoint、offboard/armed/control mode；如果控制器发布 attitude setpoint，也要记录。
- 感知结果：handle pose、检测置信度、检测是否有效；RGB/depth 原始图像建议用 rosbag 另存，用于论文中的定性图和失败案例分析。
- 任务几何：payload pose、handle pose、dock pose、tank bounds。仿真中可来自 Gazebo ground truth，真实水池中可来自 mocap、标定点或人工测量的 metadata。
- 任务事件：`start`、`first_detection`、`approach_start`、`hook_attempt`、`hooked`、`return_start`、`docked`、`success` 或 `failure`。
- 实验 metadata：控制器名称、参数文件、环境、payload 质量/形状、hook 版本、相机标定版本、水体/光照条件、操作者和备注。

这些数据可以支撑的论文指标：

- 任务成功率、总耗时、各阶段耗时。
- 首次检测时间、检测可用率、检测置信度统计。
- 接近精度：到 handle 或 payload 的最小/最终距离。
- 返航精度：最终 docking error 或到交付位置的距离。
- 运动质量：路径长度、平均/RMS/最大速度、最大角速度。
- 控制代价：归一化 thrust/torque 的 RMS、峰值和时间积分。
- 安全性：给定 tank bounds 时的最小边界距离和越界样本数。

论文写作时，建议对每组重复实验报告 mean、standard deviation、median，并单独讨论失败案例。

## 目录结构

- `config/trial_metadata_template.json`：每次实验前复制或修改的 metadata 模板。
- `data/`：占位目录；默认运行日志写到 `/home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials`。
- `../../bluerov2_control/research/trial_data_logger.py`：ROS 2 CSV 数据采集节点。
- `../../bluerov2_control/research/mark_trial_event.py`：人工事件标记命令。
- `../../bluerov2_control/research/analyze_trial.py`：离线分析脚本。

## 运行数据采集

先构建并 source workspace：

```bash
cd ~/bluerov_ws
colcon build --packages-select bluerov2_control
source install/setup.bash
```

## 记录水池固定挂钩目标

启动 MoCap EKF 后，让机器人保持在“工具已经勾住箱子把手”的姿态，直接从
`/mocap/glub/odom_ekf` 自动平均记录目标位姿：

```bash
ros2 run bluerov2_control record_mocap_target_pose \
  --topic /mocap/glub/odom_ekf \
  --message-type odom \
  --samples 80 \
  --output-file /home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/hooked_box_target_pose_from_ekf.json
```

这会生成一个新的 target JSON。后续 fixed-target 验证和数据记录时，优先使用这个
从 EKF topic 直接记录的文件，而不是手动从终端截图抄数值。

## 自动 fixed-hook MPC 测试

当前 June 23 基准已经接入 MPC trajectory tracking。确认 MoCap 软件在实验室电脑上
稳定发布 `/mocap/glub/pose` 后，先标定并验证修正后的 EKF。

如果重新贴了 marker、重新定义了 Qualisys rigid body，或者发现 corrected EKF 仍然
因为 `base_link_z_axis_angle` 被拒绝，先让机器人实际保持水平静止，然后运行：

```bash
ros2 run bluerov2_control calibrate_mocap_orientation_correction \
  --topic /mocap/glub/pose \
  --samples 120
```

命令会输出新的 `orientation_correction_quat_xyzw`。把输出值用于下面的 EKF 测试和
fixed-hook launch。

终端 A：

```bash
ros2 launch bluerov2_control mocap_ekf_odom.launch.py \
  rigid_body_name:=glub \
  orientation_correction_quat_xyzw:="0.04430086214711217 -0.001252015250325171 -5.551990616173286e-05 0.9990174487907484"
```

终端 B：

```bash
ros2 topic hz /mocap/glub/pose
ros2 topic hz /mocap/glub/odom_ekf
ros2 topic hz /mocap/glub/vehicle_odometry_ekf
```

验证稳定后，停止终端 A 的 standalone EKF，再启动完整 fixed-hook MPC：

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py
```

该 launch 会启动 MoCap EKF、`nav_msgs/Odometry` 到 `px4_msgs/VehicleOdometry`
的适配器、offboard heartbeat、MPC trajectory tracking 和数据记录，并会自动给
MPC 进程设置 `/home/yecheng/acados` 相关的 `ACADOS_SOURCE_DIR` 和
`LD_LIBRARY_PATH`。MPC 内部带有 odometry 超时保护；如果 MoCap/EKF odometry
超过 0.30 s 没有更新，会发布零 thrust/torque。

`fixed_hook_mpc_june23.launch.py` 默认会对 `glub` 的 MoCap orientation 应用固定
旋转修正，并把同一修正应用到旧 target JSON 的 orientation 上。当前修正来自
2026-06-28 水平静止标定，修正后 base_link z-axis angle 最大约 0.34 deg；如果重新
在 Qualisys 中定义了 rigid body 坐标轴，需要重新标定或将
`orientation_correction_quat_xyzw` 改回空字符串。

第一次真实水池测试保持低限幅：

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py \
  thrust_sat:=0.04 \
  torque_sat:=0.05
```

如果出现持续转圈或 MoCap warning，立即停止 launch，先不要提高限幅。

如果真实 PX4 topic 使用 `/itrl_rov_1` 而不是 `/glub`：

```bash
ros2 launch bluerov2_control fixed_hook_mpc_june23.launch.py \
  robot_namespace:=/itrl_rov_1
```

普通 `/itrl_rov_1` 仿真实验：

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=stabilized_control \
  environment:=gazebo_tank \
  metadata_file:=/home/yecheng/bluerov_ws/src/bluerov2_control/experiments/payload_retrieval/config/trial_metadata_template.json
```

水池实验并记录 mocap：

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  controller_name:=mpc_track_trajectory_acados \
  environment:=kth_pool \
  mocap_odom_topic:=/mocap/itrl_rov_1/odom
```

如果当前 topic 名称不同，可以在 launch 时覆盖：

```bash
ros2 launch bluerov2_control payload_retrieval_data_collection.launch.py \
  odom_topic:=/fmu/out/vehicle_odometry \
  thrust_sp_topic:=/fmu/in/vehicle_thrust_setpoint \
  torque_sp_topic:=/fmu/in/vehicle_torque_setpoint \
  control_mode_topic:=/fmu/out/vehicle_control_mode
```

从另一个终端标记关键事件：

```bash
ros2 run bluerov2_control mark_trial_event start --note "mission started"
ros2 run bluerov2_control mark_trial_event first_detection
ros2 run bluerov2_control mark_trial_event hook_attempt
ros2 run bluerov2_control mark_trial_event hooked
ros2 run bluerov2_control mark_trial_event docked
ros2 run bluerov2_control mark_trial_event success
```

每次运行会生成一个 trial 目录，包含：

- `metadata.json`
- `samples.csv`
- `events.csv`

## 离线分析

分析一个 trial：

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS
```

如果实验中没有发布 payload/dock pose，可以手动传入静态目标点：

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --payload-target -1.0,-2.0,95.7 \
  --dock-target 0.0,0.0,95.7
```

如果需要安全边界分析，传入 tank bounds：

```bash
ros2 run bluerov2_control analyze_payload_retrieval_trial \
  /home/yecheng/bluerov_ws/bluerov2_payload_retrieval_trials/retrieval_YYYYMMDD_HHMMSS \
  --tank-bounds -4.5,4.5,-2.5,2.5,94.2,97.2
```

分析结果会写入 trial 目录下的 `analysis/`：

- `summary_metrics.json`
- `summary_metrics.csv`
- `thesis_results_summary.md`
- 如果安装了 `matplotlib`，还会生成 plan view、目标距离和控制量图。
