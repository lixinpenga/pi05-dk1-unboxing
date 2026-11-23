# 贡献指南

欢迎提交可复现的错误报告、文档改进和 DK1 工程验证。先检查[现有 Issues](https://github.com/lixinpenga/pi05-dk1-unboxing/issues)。

## 提交变更

1. Fork 后创建分支，按 [README](README.md) 安装开发依赖。
2. 运行 `ruff check .`、`pytest -q` 和 `dk1-unboxing smoke --output outputs/smoke-contrib`。
3. 涉及 LeRobot v3、数据字段或动作时间轴时，再运行 `dk1-unboxing smoke --lerobot --output outputs/smoke-lerobot-contrib`。
4. PR 中写出影响的命令、验证环境和结果。若变更会改变真机行为，说明测试时的控制频率、夹爪与工位配置，以及是否在实体设备验证。

## 报告实验与故障

附上命令、commit SHA、Python / LeRobot 版本、数据或模型修订，以及最小日志或样本。真机问题请一并给出机型、相机设置、RTC 参数、发生阶段和动作队列日志。提交前移除私有日志中的账号、令牌、设备序列号和未授权媒体。

上传新的实验素材时请附上记录场景、实验条件和拍摄来源。
