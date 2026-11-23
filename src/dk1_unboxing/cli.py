"""Command-line entry points for data collection, training and robot evaluation."""

import argparse
import json
import shlex
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from uuid import uuid4

import yaml

from .checkpoints import fetch_model
from .dataset import export, inspect
from .evaluation import summarize
from .recording import Episode, write_json
from .rtc import PRESETS, RTCSettings
from .scene import SCENES, TASK, Scene
from .smoke import offline_smoke
from .training import Recipe, build_train_args, launch, launch_command
from .validation import episode_split, estimate_action_error


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_train(args):
    recipes = load(args.recipes)
    values = dict(recipes[args.recipe])
    for field in fields(Recipe):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    recipe = Recipe(**values)
    if args.device == "cpu" and recipe.gpus != 1:
        raise ValueError("CPU 调试仅支持单进程")
    if args.dataset_root and not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    revision = args.revision
    episodes = None
    eval_fraction = None
    if not args.dry_run:
        info = inspect(recipe.dataset, args.dataset_root, revision)
        if recipe.eval_every:
            if info["total_tasks"] != 1:
                raise ValueError("当前训练配方的回合留出适用于单任务数据")
            train_episodes, valid = episode_split(info["total_episodes"], args.dataset_root)
            episodes = train_episodes + valid
            # LeRobot holds out ceil(fraction × episode count), starting at the end.
            eval_fraction = (len(valid) - 0.5) / len(episodes)
        if args.device == "cuda":
            import torch

            if torch.cuda.device_count() < recipe.gpus:
                raise RuntimeError(f"训练需要 {recipe.gpus} 个可见 CUDA GPU")
    init = args.init
    init_revision = args.init_revision
    options = build_train_args(
        recipe,
        args.output,
        init,
        args.dataset_root,
        revision,
        args.device,
        init_revision,
        episodes,
        eval_fraction,
    )
    command = launch_command(recipe.gpus, options)
    print(
        f"GPU={recipe.gpus} × microbatch={recipe.microbatch} × GA={recipe.accumulation}"
        f" = EBS {recipe.ebs}，优化器更新 {recipe.updates} 次，循环 {recipe.updates * recipe.accumulation} 次",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        launch(command)


def run_resume(args):
    config = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    stored = config["accelerator"]["gradient_accumulation"]["steps"]
    print(f"沿用 checkpoint 的累积步数 {stored}，请使用原有 GPU 数。", flush=True)
    options = [f"--config_path={args.checkpoint}", "--resume=true"]
    launch(launch_command(args.gpus, options))


def confirm_enable():
    text = "接通从臂会使能电机并可能校准夹爪。清空工作区并准备物理急停，输入 ARM 继续："
    if input(text).strip() != "ARM":
        raise ValueError("已取消从臂连接")


def run_record(args):
    from .hardware import LiveRobot, make_teleop
    from .live import connected_teleop, record_demo

    config = load(args.hardware)
    slot = Scene(**load(args.scene_config)).slot(args.slot)
    emit(slot)
    episode = Episode(
        args.id or uuid4().hex,
        args.kind,
        args.scene,
        args.operator,
        args.teleop,
        args.slot,
        0,
        task=args.task,
        round=args.round,
        source_offset_cm=tuple(args.source_offset),
        tray_offset_cm=tuple(args.tray_offset),
    )
    robot = LiveRobot(config)
    teleop = make_teleop(args.teleop, config)
    confirm_enable()
    with robot, connected_teleop(teleop):
        print(record_demo(robot, teleop, args.raw, episode))


def run_rollout(args):
    from .hardware import LiveRobot, make_teleop
    from .live import connected_teleop, rollout
    from .policy import Pi05Predictor

    config = load(args.hardware)
    plan = load(args.plan)
    round_info = next((trial for trial in plan["trials"] if trial["round"] == args.round), None)
    if round_info is None:
        raise ValueError(f"计划不包含第 {args.round} 轮")
    slots = [slot for slot in round_info["slots"] if args.slot is None or slot["index"] == args.slot]
    if not slots:
        raise ValueError("计划中没有指定的工件")
    settings = PRESETS[args.rtc]
    settings = RTCSettings(
        reinfer=args.reinfer if args.reinfer is not None else settings.reinfer,
        blend=args.blend if args.blend is not None else settings.blend,
        initial_delay=args.delay if args.delay is not None else settings.initial_delay,
        guided=settings.guided,
    )
    predictor = Pi05Predictor(args.checkpoint, args.device, args.task, settings)
    robot = LiveRobot(config)
    teleop = make_teleop(args.teleop, config) if args.raw else None
    confirm_enable()
    with ExitStack() as stack:
        stack.enter_context(robot)
        if teleop is not None:
            stack.enter_context(connected_teleop(teleop))
        for slot in slots:
            id_ = f"round{args.round:02d}-slot{slot['index']:02d}"
            result_path = args.output / f"{id_}.json"
            if result_path.exists() and args.slot is None:
                prior = load(result_path)
                if prior["verdict"]["reason"] in {"operator_abort", "hardware_fault"}:
                    raise RuntimeError(f"{result_path} 中有中止记录，请归档并复位后复测")
                continue
            emit({**slot, **{key: round_info[key] for key in ("source_offset_cm", "tray_offset_cm")}})
            episode = Episode(
                f"{id_}-{uuid4().hex[:8]}",
                "dagger" if args.raw else "rollout",
                plan["scene_name"],
                args.operator,
                args.teleop if args.raw else "none",
                slot["index"],
                0,
                task=args.task,
                round=args.round,
                source_offset_cm=tuple(round_info["source_offset_cm"]),
                tray_offset_cm=tuple(round_info["tray_offset_cm"]),
            )
            result = rollout(
                robot,
                predictor,
                settings,
                episode,
                result_path,
                args.raw,
                teleop,
                str(args.checkpoint.resolve()),
            )
            emit({"verdict": result["verdict"], "inference_p95_ms": result["inference_p95_ms"]})
            if result["fault"] or result["verdict"]["reason"] == "operator_abort":
                raise RuntimeError(result["fault"] or "操作员已中止，停止本轮")


def run_report(args):
    results = [load(path) for path in sorted(args.results.glob("round*-slot*.json"))]
    report = summarize(load(args.plan), results)
    if args.output:
        write_json(args.output, report)
    emit(report)


def main():
    parser = argparse.ArgumentParser(description="π0.5 × TRLC-DK1 执行器拆箱")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="离线跑通记录、RTC 和评分")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--lerobot", action="store_true", help="同时导出并读回 LeRobot v3 数据")
    smoke.set_defaults(func=lambda args: emit(offline_smoke(args.output, args.lerobot)))

    plan = sub.add_parser("plan", help="生成固定顺序的 40 件评测计划")
    plan.add_argument("--scene-config", type=Path, default=Path("configs/scene.yaml"))
    plan.add_argument("--rounds", type=int, default=1)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--diverse", action="store_true")
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(
        func=lambda args: write_json(
            args.output,
            Scene(**load(args.scene_config)).plan(args.rounds, args.seed, args.diverse),
        )
    )

    inspect_data = sub.add_parser("inspect-data", help="读取公开或本地 LeRobot 数据结构")
    data_source = inspect_data.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--dataset", help="Hugging Face 数据集 ID")
    data_source.add_argument("--root", type=Path)
    inspect_data.add_argument("--revision")
    inspect_data.set_defaults(func=lambda args: emit(inspect(args.dataset, args.root, args.revision)))

    model = sub.add_parser("fetch-model", help="按固定版本下载公开权重和处理器")
    model.add_argument("model", help="Hugging Face 模型仓库 ID")
    model.add_argument("--revision", help="可选的分支、标签或提交 SHA")
    model.add_argument("--output", type=Path)
    model.add_argument("--metadata-only", action="store_true")
    model.set_defaults(
        func=lambda args: print(
            fetch_model(args.model, args.output, args.metadata_only, revision=args.revision)
        )
    )

    export_data = sub.add_parser("export", help="裁剪、过滤并生成 LeRobot v3 数据")
    export_data.add_argument("--raw", type=Path, required=True)
    export_data.add_argument("--output", type=Path, required=True)
    export_data.add_argument("--repo", default="local/dk1-unboxing")
    export_data.add_argument("--route", choices=["baseline", "hq", "dagger"], default="baseline")
    export_data.add_argument("--idle-threshold", type=float, default=0.004)
    export_data.add_argument("--idle-margin", type=int, default=5)
    export_data.set_defaults(
        func=lambda args: emit(
            export(
                args.raw,
                args.output,
                args.repo,
                args.route,
                args.idle_threshold,
                args.idle_margin,
            )
        )
    )

    train = sub.add_parser("train", help="根据训练配方启动 π0.5 微调")
    train.add_argument("recipe")
    train.add_argument("--recipes", type=Path, default=Path("configs/recipes.yaml"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--init", required=True, help="初始 checkpoint 路径或 Hugging Face 模型仓库 ID")
    train.add_argument("--init-revision")
    train.add_argument("--dataset-root", type=Path)
    train.add_argument("--revision")
    train.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--dataset", required=True, help="LeRobot 数据集 ID")
    for name in (
        "updates",
        "ebs",
        "microbatch",
        "gpus",
        "warmup",
        "decay",
        "eval_every",
        "save_every",
        "workers",
    ):
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    train.set_defaults(func=run_train)

    resume = sub.add_parser("resume", help="从本项目保存的 checkpoint 恢复训练状态")
    resume.add_argument(
        "--checkpoint", type=Path, required=True, help="pretrained_model/train_config.json"
    )
    resume.add_argument("--gpus", type=int, required=True)
    resume.set_defaults(func=run_resume)

    record = sub.add_parser("record", help="用 Gello 或 Quest 在 DK1 上录制一个示教")
    record.add_argument("--scene-config", type=Path, default=Path("configs/scene.yaml"))
    record.add_argument("--scene", choices=SCENES, default="lightbox")
    record.add_argument("--slot", type=int, required=True)
    record.add_argument("--round", type=int, default=0)
    record.add_argument("--kind", choices=["demo", "hq"], default="demo")
    record.add_argument("--source-offset", nargs=2, type=float, default=(0, 0))
    record.add_argument("--tray-offset", nargs=2, type=float, default=(0, 0))
    record.add_argument("--id")
    record.add_argument("--raw", type=Path, required=True)
    record.set_defaults(func=run_record)

    run = sub.add_parser("rollout", help="用 RTC 在真机上执行，可启用人工接管")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--round", type=int, default=0)
    scope = run.add_mutually_exclusive_group(required=True)
    scope.add_argument("--slot", type=int)
    scope.add_argument("--all", action="store_true", help="逐个确认并执行该轮所有工件")
    run.add_argument("--rtc", choices=PRESETS, default="baseline")
    run.add_argument("--reinfer", type=int)
    run.add_argument("--blend", type=int)
    run.add_argument("--delay", type=int)
    run.add_argument("--device", default="cuda")
    run.add_argument("--raw", type=Path, help="写入实际动作并开启 DAgger 人工接管")
    run.set_defaults(func=run_rollout)

    for live in (record, run):
        live.add_argument("--hardware", type=Path, required=True)
        live.add_argument("--operator", required=True)
        live.add_argument("--teleop", choices=["gello", "vr"], default="gello")
        live.add_argument("--task", default=TASK)

    val = sub.add_parser("offline-eval", help="在留出的回合上生成动作块并计算有效时步的 L2")
    val.add_argument("--checkpoint", type=Path, required=True)
    val.add_argument("--dataset", required=True, help="LeRobot 数据集 ID")
    val.add_argument("--dataset-root", type=Path)
    val.add_argument("--revision")
    val.add_argument("--max-anchors", type=int, default=128)
    val.add_argument("--device", default="cuda")
    val.set_defaults(
        func=lambda args: emit(
            estimate_action_error(
                args.checkpoint,
                args.dataset,
                args.dataset_root,
                args.revision,
                args.max_anchors,
                args.device,
            )
        )
    )

    report = sub.add_parser("report", help="汇总阶段成功率、试验覆盖率及 Wilson 区间")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
