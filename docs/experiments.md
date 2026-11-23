# 实验手册

[项目首页](../README.md) · [训练细节](training.md) · [真机手册](hardware.md)

## 实验入口

| 目标 | 所需资源 | 输出 |
| --- | --- | --- |
| 验证代码链路 | CPU、Python 3.12，可选 LeRobot | 合成回合的导出、回读与 RTC 时序报告 |
| 运行全参训练或小数据路线 | CUDA GPU、LeRobot 数据 | 本地 checkpoint、训练和离线验证日志 |
| 复测 RTC 与最终成功率 | 已校准 DK1 双臂、三台相机、现场操作者和 GPU | 单回合结果、每轮成功率、阶段直方图、Wilson 区间 |

离线入口见[快速开始](../README.md#快速开始)。若 CPU 运行了 LeRobot 冒烟测试，报告中的回合和控制时序来自合成输入。

## 从示教到训练

训练与真机依赖按[训练手册](training.md)和[真机手册](hardware.md)安装。校准后用 Quest 或 Gello 录制。下面以 Gello 的高质量示教为例，录制完成后按终端提示确认保存和评级。

```bash
dk1-unboxing record --hardware configs/hardware.local.yaml \
  --teleop gello --operator op01 --slot 0 --kind hq --raw data/raw
dk1-unboxing export --raw data/raw --output data/hq --route hq --repo local/dk1-hq

dk1-unboxing train hq --dataset local/dk1-hq --dataset-root data/hq \
  --init weights/initial --output outputs/hq --dry-run
```

若使用自采 HQ 数据训练，重复采集覆盖不同件位的多个回合，再给 `train hq` 加上 `--dataset local/dk1-hq --dataset-root data/hq`。单条示教用于检查录制和导出链路。基线训练在一台机器上用四卡，每卡 microbatch 8、梯度累积 8、EBS 256，运行 50,000 次优化器更新。短跑、断点恢复、离线动作误差及相对动作消融见[训练手册](training.md)。

## 真机对照两组 RTC 参数

在相同 checkpoint、工位摆放与测试计划上比较。每套参数使用独立结果目录。配置好设备、关节和夹爪限值后先验证一个件位，再扩大到全轮。

```bash
dk1-unboxing plan --rounds 1 --output outputs/lightbox-plan.json

dk1-unboxing rollout --hardware configs/hardware.local.yaml \
  --checkpoint outputs/hq/checkpoints/last/pretrained_model --plan outputs/lightbox-plan.json \
  --round 0 --slot 0 --rtc baseline --operator op01 --output outputs/rig-baseline

dk1-unboxing rollout --hardware configs/hardware.local.yaml \
  --checkpoint outputs/hq/checkpoints/last/pretrained_model --plan outputs/lightbox-plan.json \
  --round 0 --slot 0 --rtc fast --operator op01 --output outputs/rig-fast

dk1-unboxing report --plan outputs/lightbox-plan.json --results outputs/rig-fast
```

完成同一轮剩余件位时把 `--slot 0` 改成 `--all`。在现场按计划恢复场景，逐件确认。评测保存模型路径、RTC 参数、完成阶段、终止原因和推理延迟。报告保留样本分母、未完成件位、排除原因与置信区间。
