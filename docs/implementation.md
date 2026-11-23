# 实现说明

| 环节 | 配置与实现 |
| --- | --- |
| 工位 | 每轮评测顶层 40 件，目标托盘为 9×15。取件和填盘依照配置的格位顺序。 |
| 采集 | 顶视、左腕和右腕三路 640×480 MJPG 图像，控制循环 50 Hz。V4L2 按主机接收时间记录，导出时中心裁剪至 224×224。 |
| 示教 | 支持 Gello 与 Quest。按质量、空闲帧与控制权来源筛选示教回合和干预片段，导出为 LeRobot v3。 |
| 训练 | π₀.₅ 基线采用单机多卡 DDP、BF16、AdamW 与 WSD 调度。默认 EBS 256，共 50,000 次优化器更新。 |
| 部署 | Guided RTC 使用 10 步流积分。在控制周期中按时间戳丢弃过期动作，并记录推理延迟。 |
| 评测 | 40 件为一轮，单件 45 秒上限，10 秒无进展提前结束。记录五阶段得分、终止原因、实际分母和 Wilson 区间。 |

部署时使用 checkpoint 保存的分位数统计。LeRobot 的加载钩子负责迁移视觉权重字段。更换工件、腕相机视野或双臂位置后，应重新校准并评测。

[工位与采集](hardware.md) · [训练配置](training.md) · [公开模型与结果](experiments.md)

## 依赖项目

- [LeRobot](https://github.com/huggingface/lerobot)
- [TRLC-DK1](https://github.com/robot-learning-co/trlc-dk1)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
