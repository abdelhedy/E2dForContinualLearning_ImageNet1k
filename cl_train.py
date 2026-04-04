"""
cl_train.py
-----------
Main entry point for E2D-based continual learning on Split-ImageNet-1K.

Two strategies compared:
  RandomReplay  — Avalanche's built-in reservoir sampling replay
  E2DReplay     — E2D dataset distillation + soft-label KD (this work)

Timing is recorded per strategy: wall-clock start / end time and total seconds
are printed in the summary table.

Training recipe (faithful to train_FKD_parallel.py):
  • SGD + cosine LR schedule (matches the paper's student schedule)
  • DIST loss for replay KD (default)
  • Per-experience LR reset so each task trains with a fresh schedule
"""

import argparse
import math
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from cl_plugin import E2DReplayPlugin


# ══════════════════════════════════════════════════════════════════════
#  LR scheduler plugin: cosine + per-experience reset
#  (mirrors the student LR policy from train_FKD_parallel.py)
# ══════════════════════════════════════════════════════════════════════

class CosineLRPlugin(SupervisedPlugin):
    """
    Resets a cosine-annealing schedule at the start of each experience.
    Mirrors the 'cos' ls-type from train_FKD_parallel.py.
    """

    def __init__(self, total_epochs: int, eta_min: float = 0.0):
        super().__init__()
        self.total_epochs = int(total_epochs)
        self.eta_min      = float(eta_min)
        self.scheduler    = None

    def before_training_exp(self, strategy, **kwargs) -> None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            strategy.optimizer,
            T_max=self.total_epochs,
            eta_min=self.eta_min,
        )
        lr = strategy.optimizer.param_groups[0]["lr"]
        exp_id = strategy.experience.current_experience
        print(f"[CosineLR] Reset for exp {exp_id} | start lr={lr:.6f} | T_max={self.total_epochs}")

    def after_training_epoch(self, strategy, **kwargs) -> None:
        if self.scheduler is not None:
            self.scheduler.step()
            lr = strategy.optimizer.param_groups[0]["lr"]
            print(f"[CosineLR] epoch {strategy.clock.train_exp_epochs} → lr={lr:.6f}")


# ══════════════════════════════════════════════════════════════════════
#  Benchmark: Split-ImageNet-1K
# ══════════════════════════════════════════════════════════════════════

def build_imagenet_benchmark(
    imagenet_path: str,
    n_experiences: int,
    seed: int,
    num_classes: int = 1000,
):
    """
    Use Avalanche's nc_benchmark to split ImageNet-1K into `n_experiences`
    disjoint class groups (New Classes scenario).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = ImageNet(root=imagenet_path, split="train", transform=train_transform)
    val_ds   = ImageNet(root=imagenet_path, split="val",   transform=val_transform)

    return nc_benchmark(
        train_dataset=train_ds,
        test_dataset=val_ds,
        n_experiences=n_experiences,
        shuffle=False,   # deterministic class split
        seed=seed,
        task_labels=False,
    )


# ══════════════════════════════════════════════════════════════════════
#  Student model
# ══════════════════════════════════════════════════════════════════════

def build_student(arch: str, num_classes: int) -> nn.Module:
    model = tv_models.__dict__[arch](weights=None)
    # Replace the final FC for the full class count
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        # e.g. VGG / EfficientNet
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


# ══════════════════════════════════════════════════════════════════════
#  Optimizer (faithful to train_FKD_parallel.py SGD recipe)
# ══════════════════════════════════════════════════════════════════════

def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.875,
        weight_decay=3e-5,
        nesterov=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  Evaluator
# ══════════════════════════════════════════════════════════════════════

def build_evaluator() -> EvaluationPlugin:
    return EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )


# ══════════════════════════════════════════════════════════════════════
#  Strategy runner with timing
# ══════════════════════════════════════════════════════════════════════

def run_strategy(strategy, benchmark, name: str) -> dict:
    print(f"\n{'=' * 65}\n  Strategy: {name}\n{'=' * 65}")

    t_start   = time.time()
    ts_start  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Timing] {name} started at {ts_start}")

    results = []
    for experience in benchmark.train_stream:
        exp_id = experience.current_experience
        cls    = experience.classes_in_this_experience
        print(f"\n--- Exp {exp_id} | classes {cls} ---")
        strategy.train(experience)
        results.append(strategy.eval(benchmark.test_stream))

    t_end    = time.time()
    ts_end   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed  = t_end - t_start

    print(f"[Timing] {name} ended at   {ts_end}")
    print(f"[Timing] {name} total time: {elapsed/3600:.2f} h  ({elapsed:.0f} s)")

    final   = results[-1]
    acc_key = next((k for k in final if "Top1_Acc_Stream"    in k), None)
    fgt_key = next((k for k in final if "StreamForgetting"   in k), None)

    return {
        "name":         name,
        "start":        ts_start,
        "end":          ts_end,
        "time_seconds": elapsed,
        "time_h":       f"{elapsed / 3600:.2f}h",
        "final_acc":    final[acc_key] * 100 if acc_key else 0.0,
        "forgetting":   final[fgt_key] * 100 if fgt_key else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E2D Continual Learning on Split-ImageNet-1K"
    )

    # ── Data paths ───────────────────────────────────────────────────
    parser.add_argument("--imagenet-path", type=str, required=True,
                        help="Path to ImageNet root (contains train/ and val/)")
    parser.add_argument("--output-dir",    type=str, default="./e2d_cl_output",
                        help="Root for synthetic images and soft labels")
    parser.add_argument("--statistic-path",type=str, default="./statistic",
                        help="BN statistic cache (shared across all tasks)")
    parser.add_argument("--recover-script",type=str, default="./recover_cl.py",
                        help="Path to recover_cl.py")

    # ── CL scenario ──────────────────────────────────────────────────
    parser.add_argument("--n-experiences", type=int, default=10,
                        help="Number of CL tasks (100 classes / task for 1K)")
    parser.add_argument("--num-classes",   type=int, default=1000)
    parser.add_argument("--seed",          type=int, default=42)

    # ── Student ──────────────────────────────────────────────────────
    parser.add_argument("--student-arch",  type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "mobilenet_v3_large"])
    parser.add_argument("--epochs",        type=int, default=100,
                        help="Training epochs per experience")
    parser.add_argument("--lr",            type=float, default=1.024,
                        help="SGD initial LR (train_FKD_parallel.py default)")
    parser.add_argument("--batch-size",    type=int, default=1024)

    # ── Strategy ─────────────────────────────────────────────────────
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "random", "e2d"])
    parser.add_argument("--no-cuda",  action="store_true")

    # ── Buffer ───────────────────────────────────────────────────────
    parser.add_argument("--ipc",             type=int,  default=50,
                        help="Images per class in the E2D replay buffer")
    parser.add_argument("--fixed-per-class", action="store_true", default=True)

    # ── Recover hyper-params ─────────────────────────────────────────
    parser.add_argument("--recover-iterations",  type=int,   default=1000)
    parser.add_argument("--recover-lr",          type=float, default=0.1)
    parser.add_argument("--recover-batch-size",  type=int,   default=100)
    parser.add_argument("--K",                   type=int,   default=700,
                        help="Exploration→exploitation switch iteration")
    parser.add_argument("--loss-threshold",      type=float, default=0.5)
    parser.add_argument("--r-loss",              type=float, default=0.05)
    parser.add_argument("--first-multiplier",    type=float, default=10.0)
    parser.add_argument("--tv-l2",               type=float, default=0.0001)
    parser.add_argument("--training-momentum",   type=float, default=0.4)
    parser.add_argument("--gpu-id",              type=str,   default="0",
                        help="GPU(s) visible to recover_cl.py subprocess")

    # ── Relabeling ───────────────────────────────────────────────────
    parser.add_argument("--relabel-views",       type=int,   default=10,
                        help="Number of augmented views per image for soft-label averaging")
    parser.add_argument("--relabel-temperature", type=float, default=20.0,
                        help="Softmax temperature during relabeling (matches generate_soft_label paper default)")
    parser.add_argument("--relabel-batch-size",  type=int,   default=64)

    # ── Replay KD ────────────────────────────────────────────────────
    parser.add_argument("--kd-loss",        type=str,   default="dist",
                        choices=["kl", "dist", "mse_gt"])
    parser.add_argument("--kd-weight",      type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    torch.manual_seed(args.seed)
    print(f"Device       : {device}")
    print(f"Student arch : {args.student_arch}")
    print(f"Experiences  : {args.n_experiences} ({args.num_classes // args.n_experiences} classes each)")
    print(f"IPC          : {args.ipc}  |  fixed-per-class: {args.fixed_per_class}")

    benchmark = build_imagenet_benchmark(
        args.imagenet_path, args.n_experiences, args.seed, args.num_classes
    )

    common_plugins = [CosineLRPlugin(total_epochs=args.epochs)]
    results_table  = []

    # ── Strategy 1: Random Replay ─────────────────────────────────────
    if args.strategy in ("all", "random"):
        mem_size = (
            args.ipc * args.num_classes
            if args.fixed_per_class
            else args.ipc
        )
        model     = build_student(args.student_arch, args.num_classes).to(device)
        optimizer = make_optimizer(model, args)
        strategy  = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
            plugins=list(common_plugins) + [ReplayPlugin(mem_size=mem_size)],
        )
        results_table.append(run_strategy(strategy, benchmark, "RandomReplay"))

    # ── Strategy 2: E2D Replay ────────────────────────────────────────
    if args.strategy in ("all", "e2d"):
        e2d_plugin = E2DReplayPlugin(
            output_dir=args.output_dir,
            recover_script=args.recover_script,
            train_data_path=str(Path(args.imagenet_path) / "train"),
            statistic_path=args.statistic_path,
            ipc=args.ipc,
            fixed_per_class=args.fixed_per_class,
            num_classes=args.num_classes,
            device=device,
            recover_iterations=args.recover_iterations,
            recover_lr=args.recover_lr,
            recover_batch_size=args.recover_batch_size,
            K=args.K,
            loss_threshold=args.loss_threshold,
            r_loss=args.r_loss,
            first_multiplier=args.first_multiplier,
            tv_l2=args.tv_l2,
            training_momentum=args.training_momentum,
            gpu_id=args.gpu_id,
            relabel_views=args.relabel_views,
            relabel_temperature=args.relabel_temperature,
            relabel_batch_size=args.relabel_batch_size,
            kd_loss=args.kd_loss,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
        )
        model     = build_student(args.student_arch, args.num_classes).to(device)
        optimizer = make_optimizer(model, args)
        strategy  = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
            plugins=list(common_plugins) + [e2d_plugin],
        )
        results_table.append(run_strategy(strategy, benchmark, "E2DReplay"))

    # ── Summary ───────────────────────────────────────────────────────
    if results_table:
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        header = f"{'Strategy':>14}  {'Acc%':>7}  {'Forget%':>9}  {'Time':>8}  Start → End"
        print(header)
        print("-" * 65)
        for r in results_table:
            print(
                f"{r['name']:>14}  "
                f"{r['final_acc']:>7.2f}  "
                f"{r['forgetting']:>9.2f}  "
                f"{r['time_h']:>8}  "
                f"{r['start']}  →  {r['end']}"
            )
        print("=" * 65)


if __name__ == "__main__":
    main()
