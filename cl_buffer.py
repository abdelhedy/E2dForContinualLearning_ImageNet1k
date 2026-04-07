"""
cl_buffer.py
------------
Disk-backed replay buffer for E2D continual learning on ImageNet.

On-disk layout (produced by recover_cl.py + cl_plugin.py):
    {root}/task_{T}/syn/new{class_id:03d}/class{class_id:03d}_id{ipc:03d}.jpg
    {root}/task_{T}/soft_labels_{class_id}.pt     ← [N, C] averaged teacher logits

The buffer keeps only path metadata in memory.
Images are loaded from disk on demand inside SyntheticReplayDataset.
Soft-label tensors are loaded once per class and cached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ─── ImageNet-standard normalisation (same as recover.py / generate_soft_label…) ───
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

REPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ──────────────────────── replay dataset ──────────────────────────────

class SyntheticReplayDataset(Dataset):
    """
    Loads distilled JPEG images from disk together with their soft labels.
    Covers all buffered classes / tasks in one unified dataset.

    Each item: (image_tensor, class_id, soft_label_vector)
    """

    def __init__(
        self,
        records: List[Tuple[str, int, torch.Tensor]],
        transform=REPLAY_TRANSFORM,
    ):
        # records[i] = (absolute_jpeg_path, class_id, soft_label [C])
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        img_path, class_id, soft_label = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return (
            img,
            torch.tensor(class_id, dtype=torch.long),
            soft_label.clone(),
        )


# ──────────────────────────── buffer ──────────────────────────────────

class E2DBuffer:
    """
    Manages paths to synthetic images and averaged soft-label tensors.

    Two budget modes:
      fixed_per_class=True  → each class always keeps `ipc` images
      fixed_per_class=False → budget is shared: ipc // total_classes
    """

    def __init__(self, root: str, ipc: int, fixed_per_class: bool = True):
        self.root = Path(root)
        self.ipc = ipc
        self.fixed_per_class = fixed_per_class

        # {class_id: {"task_id": int, "img_paths": List[str], "soft_labels": Tensor[N,C]}}
        self._data: Dict[int, dict] = {}

    # ── budget ──────────────────────────────────────────────────────

    def budget_per_class(self, total_classes: int) -> int:
        if self.fixed_per_class:
            return self.ipc
        return max(1, self.ipc // max(1, total_classes))

    # ── properties ──────────────────────────────────────────────────

    @property
    def total_images(self) -> int:
        return sum(len(v["img_paths"]) for v in self._data.values())

    @property
    def seen_classes(self) -> List[int]:
        return sorted(self._data.keys())

    # ── update ──────────────────────────────────────────────────────

    def update(
        self,
        task_id: int,
        class_id: int,
        img_paths: List[str],
        soft_labels: torch.Tensor,
        total_classes: int,
    ) -> None:
        """Register or replace a class's synthetic images and soft labels."""
        budget = self.budget_per_class(total_classes)

        # If using shared budget, trim all existing classes too
        if not self.fixed_per_class:
            for cid in list(self._data.keys()):
                if len(self._data[cid]["img_paths"]) > budget:
                    self._data[cid]["img_paths"]   = self._data[cid]["img_paths"][:budget]
                    self._data[cid]["soft_labels"] = self._data[cid]["soft_labels"][:budget]

        self._data[class_id] = {
            "task_id":    int(task_id),
            "img_paths":  [str(p) for p in img_paths[:budget]],
            "soft_labels": soft_labels[:budget].detach().cpu(),
        }

        mode = "fixed" if self.fixed_per_class else "shared"
        print(
            f"  [Buffer] class {class_id:4d} → {len(self._data[class_id]['img_paths']):3d} imgs "
            f"| total {self.total_images:6d} imgs | {budget}/class ({mode})"
        )

    # ── dataset builder ──────────────────────────────────────────────

    def get_dataset(
        self,
        transform=REPLAY_TRANSFORM,
    ) -> Optional[SyntheticReplayDataset]:
        """Return a SyntheticReplayDataset covering all buffered classes, or None."""
        if not self._data:
            return None
        records: List[Tuple[str, int, torch.Tensor]] = []
        for class_id, info in self._data.items():
            for img_path, soft_label in zip(info["img_paths"], info["soft_labels"]):
                records.append((img_path, class_id, soft_label))
        return SyntheticReplayDataset(records, transform=transform)

    # ── path helpers ─────────────────────────────────────────────────

    def task_syn_dir(self, task_id: int) -> Path:
        """Root of the recover.py output for task `task_id`."""
        return self.root / f"task_{task_id}" / "syn"

    def soft_label_path(self, task_id: int, class_id: int) -> Path:
        return self.root / f"task_{task_id}" / f"soft_labels_{class_id}.pt"

    # ── persistence (resume support) ─────────────────────────────────

    def save_index(self) -> None:
        """Persist buffer metadata to disk so training can be resumed."""
        index = {
            cid: {
                "task_id":   info["task_id"],
                "img_paths": info["img_paths"],
            }
            for cid, info in self._data.items()
        }
        torch.save(index, self.root / "buffer_index.pt")

    def load_index(self) -> bool:
        """
        Try to reload buffer metadata from a previous run.
        Soft labels are re-loaded from their individual .pt files.
        Returns True if successful.
        """
        index_path = self.root / "buffer_index.pt"
        if not index_path.exists():
            return False

        index = torch.load(str(index_path))
        for cid, meta in index.items():
            task_id   = meta["task_id"]
            img_paths = meta["img_paths"]
            sl_path   = self.soft_label_path(task_id, cid)
            if not sl_path.exists():
                print(f"  [Buffer] WARNING: soft label file missing for class {cid}: {sl_path}")
                continue
            soft_labels = torch.load(str(sl_path))
            self._data[int(cid)] = {
                "task_id":    task_id,
                "img_paths":  img_paths,
                "soft_labels": soft_labels,
            }
        print(f"[Buffer] Resumed: {len(self._data)} classes, {self.total_images} images.")
        return True

    # ── repr ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        counts = {cid: len(v["img_paths"]) for cid, v in self._data.items()}
        mode = "fixed" if self.fixed_per_class else "shared"
        return (
            f"E2DBuffer(ipc={self.ipc}, mode={mode}, "
            f"classes={len(self._data)}, total={self.total_images}, "
            f"counts={counts})"
        )
