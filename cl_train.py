"""
cl_train.py
-----------
E2D Continual Learning on Split-ImageNet-1K (or Split-TinyImageNet).

Dataset choice
──────────────
  --dataset imagenet      SplitImageNet via Avalanche's nc_benchmark.
                          Requires ImageNet downloaded at --imagenet-path.
                          Avalanche does NOT bundle ImageNet; you must supply
                          the local path (train/ and val/ directories).

  --dataset tiny          SplitTinyImageNet via Avalanche's built-in benchmark.
                          Downloaded automatically (~240 MB).
                          Use this for fast iteration / no-download development.

Two strategies
──────────────
  RandomReplay  — Avalanche ReplayPlugin (reservoir sampling)
  E2DReplay     — recover_cl.py synthesis + averaged soft-label KD

Timing
──────
  Wall-clock start / end / total seconds recorded per strategy and shown
  in the final summary table.
"""

import argparse
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

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
#  LR scheduler: cosine + per-experience reset
#  mirrors 'cos' ls-type from train_FKD_parallel.py
# ══════════════════════════════════════════════════════════════════════

class CosineLRPlugin(SupervisedPlugin):
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
        lr     = strategy.optimizer.param_groups[0]["lr"]
        exp_id = strategy.experience.current_experience
        print(f"[CosineLR] Exp {exp_id} | start lr={lr:.6f} | T_max={self.total_epochs}")

    def after_training_epoch(self, strategy, **kwargs) -> None:
        if self.scheduler is not None:
            self.scheduler.step()
            lr = strategy.optimizer.param_groups[0]["lr"]
            print(f"[CosineLR] epoch {strategy.clock.train_exp_epochs} → lr={lr:.6f}")


# ══════════════════════════════════════════════════════════════════════
#  Benchmark builders
# ══════════════════════════════════════════════════════════════════════

def build_split_imagenet(imagenet_path: str, n_experiences: int, seed: int):
    """
    Split ImageNet-1K (1000 classes, 224×224) using Avalanche's nc_benchmark.

    Avalanche does not ship ImageNet. Download it from https://image-net.org
    and point --imagenet-path to the root containing train/ and val/.

    There is no workaround for the download: even Avalanche's own
    SplitImageNet wrapper (if available in future versions) still requires
    the same local files. The nc_benchmark call here is equivalent and
    explicit about what it expects.
    """
    from torchvision.datasets import ImageNet

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = ImageNet(root=imagenet_path, split="train", transform=train_tf)
    val_ds   = ImageNet(root=imagenet_path, split="val",   transform=val_tf)

    benchmark = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=val_ds,
        n_experiences=n_experiences,
        shuffle=False,   # deterministic class split keeps class→task mapping stable
        seed=seed,
        task_labels=False,
    )
    return benchmark, 1000, 224


def build_split_tiny_imagenet(n_experiences: int, seed: int,
                               dataset_root: str = "~/.avalanche/data"):
    """
    Split Tiny-ImageNet (200 classes, 64×64).
    Avalanche downloads and extracts it automatically on first run (~240 MB).
    Use --dataset tiny for fast, zero-manual-download experimentation.
    """
    try:
        from avalanche.benchmarks.classic import SplitTinyImageNet
    except ImportError:
        raise ImportError(
            "SplitTinyImageNet not available. "
            "Upgrade avalanche-lib: pip install -U avalanche-lib"
        )

    benchmark = SplitTinyImageNet(
        n_experiences=n_experiences,
        seed=seed,
        dataset_root=str(Path(dataset_root).expanduser()),
        return_task_id=False,
        shuffle=True,
    )
    return benchmark, 200, 64


# ══════════════════════════════════════════════════════════════════════
#  Student model
# ══════════════════════════════════════════════════════════════════════

def build_student(arch: str, num_classes: int, img_size: int) -> nn.Module:
    model = tv_models.__dict__[arch](weights=None)

    # Adapt stem for small images (TinyImageNet 64×64 → no aggressive downsampling)
    if img_size < 128 and hasattr(model, "conv1"):
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    # Replace classification head
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)

    return model


# ══════════════════════════════════════════════════════════════════════
#  Optimizer  (faithful to train_FKD_parallel.py SGD recipe)
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

    t0       = time.time()
    ts_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Timing] {name} started at {ts_start}")

    results = []
    for experience in benchmark.train_stream:
        exp_id = experience.current_experience
        cls    = experience.classes_in_this_experience
        print(f"\n--- Exp {exp_id} | classes {cls} ---")
        strategy.train(experience)
        results.append(strategy.eval(benchmark.test_stream))

    elapsed  = time.time() - t0
    ts_end   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Timing] {name} ended at   {ts_end}")
    print(f"[Timing] {name} total time: {elapsed / 3600:.2f}h  ({elapsed:.0f}s)")

    final   = results[-1]
    acc_key = next((k for k in final if "Top1_Acc_Stream"  in k), None)
    fgt_key = next((k for k in final if "StreamForgetting" in k), None)

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
        description="E2D Continual Learning — Split-ImageNet-1K or Split-TinyImageNet"
    )

    # ── Dataset ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="imagenet",
        choices=["imagenet", "tiny"],
        help=(
            "'imagenet': full ImageNet-1K, requires --imagenet-path (manual download). "
            "'tiny': Tiny-ImageNet-200, downloaded automatically by Avalanche."
        ),
    )
    parser.add_argument(
        "--imagenet-path", type=str, default=None,
        help="Root of ImageNet (must contain train/ and val/). Required for --dataset imagenet.",
    )
    parser.add_argument(
        "--tiny-data-root", type=str, default="~/.avalanche/data",
        help="Download/cache directory for Tiny-ImageNet (--dataset tiny).",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    parser.add_argument("--output-dir",    type=str, default="./e2d_cl_output",
                        help="Root for synthetic images and per-task soft labels")
    parser.add_argument("--recover-script",type=str, default="./recover_cl.py")

    # ── CL scenario ──────────────────────────────────────────────────
    parser.add_argument("--n-experiences", type=int, default=10)
    parser.add_argument("--seed",          type=int, default=42)

    # ── Student ──────────────────────────────────────────────────────
    parser.add_argument("--student-arch",  type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
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
                        help="Images per class in the replay buffer")
    parser.add_argument("--fixed-per-class", action="store_true", default=True)

    # ── Recover hyper-params ─────────────────────────────────────────
    parser.add_argument("--recover-iterations",  type=int,   default=1000)
    parser.add_argument("--recover-lr",          type=float, default=0.1)
    parser.add_argument("--recover-batch-size",  type=int,   default=100)
    parser.add_argument("--K",          type=int,   default=700,
                        help="Exploration→exploitation switch (E2D paper default)")
    parser.add_argument("--loss-threshold",      type=float, default=0.5)
    parser.add_argument("--r-loss",              type=float, default=0.05)
    parser.add_argument("--first-multiplier",    type=float, default=10.0)
    parser.add_argument("--tv-l2",               type=float, default=0.0001)
    parser.add_argument("--training-momentum",   type=float, default=0.4)
    parser.add_argument("--gpu-id",              type=str,   default="0",
                        help="GPU(s) for the recover_cl.py subprocess")

    # ── Relabeling ───────────────────────────────────────────────────
    parser.add_argument("--relabel-views",       type=int,   default=10,
                        help="Augmented views for soft-label averaging")
    parser.add_argument("--relabel-temperature", type=float, default=20.0,
                        help="Teacher softmax temp (generate_soft_label default)")
    parser.add_argument("--relabel-batch-size",  type=int,   default=64)

    # ── Replay KD ────────────────────────────────────────────────────
    parser.add_argument("--kd-loss",        type=str,   default="dist",
                        choices=["kl", "dist", "mse_gt"])
    parser.add_argument("--kd-weight",      type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────
    if args.dataset == "imagenet" and args.imagenet_path is None:
        parser.error(
            "--imagenet-path is required when --dataset imagenet.\n\n"
            "Avalanche (and torchvision) cannot provide ImageNet without a "
            "local copy — it is ~150 GB and requires registration at "
            "https://image-net.org.\n\n"
            "For zero-download experimentation use --dataset tiny "
            "(Tiny-ImageNet-200, ~240 MB, downloaded automatically)."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    torch.manual_seed(args.seed)

    # ── Build benchmark ───────────────────────────────────────────────
    if args.dataset == "imagenet":
        benchmark, num_classes, img_size = build_split_imagenet(
            args.imagenet_path, args.n_experiences, args.seed
        )
        train_data_path = str(Path(args.imagenet_path) / "train")
    else:
        benchmark, num_classes, img_size = build_split_tiny_imagenet(
            args.n_experiences, args.seed, args.tiny_data_root
        )
        # recover_cl.py reads real images from here for BN stats
        train_data_path = str(
            Path(args.tiny_data_root).expanduser()
            / "tiny-imagenet-200" / "train"
        )

    print(f"Dataset      : {args.dataset}  ({num_classes} classes, {img_size}×{img_size})")
    print(f"Device       : {device}")
    print(f"Student arch : {args.student_arch}")
    print(f"Experiences  : {args.n_experiences}  ({num_classes // args.n_experiences} classes each)")
    print(f"IPC          : {args.ipc}  |  fixed-per-class: {args.fixed_per_class}")

    common_plugins = [CosineLRPlugin(total_epochs=args.epochs)]
    results_table  = []

    # ── Strategy 1: Random Replay ─────────────────────────────────────
    if args.strategy in ("all", "random"):
        mem_size  = args.ipc * num_classes if args.fixed_per_class else args.ipc
        model     = build_student(args.student_arch, num_classes, img_size).to(device)
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
            train_data_path=train_data_path,  # task-filtered in recover_cl.py
            ipc=args.ipc,
            fixed_per_class=args.fixed_per_class,
            num_classes=num_classes,
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
        model     = build_student(args.student_arch, num_classes, img_size).to(device)
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

    # ── Summary table ─────────────────────────────────────────────────
    if results_table:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(
            f"{'Strategy':>14}  {'Acc%':>7}  {'Forget%':>9}  "
            f"{'Time':>8}  Start → End"
        )
        print("-" * 70)
        for r in results_table:
            print(
                f"{r['name']:>14}  "
                f"{r['final_acc']:>7.2f}  "
                f"{r['forgetting']:>9.2f}  "
                f"{r['time_h']:>8}  "
                f"{r['start']}  →  {r['end']}"
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
