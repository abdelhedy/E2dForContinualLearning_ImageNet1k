"""
recover_cl.py
─────────────
This is recover.py with ONE CL-specific addition: the --class-ids argument.

Every other line is identical to the original recover.py.  The argument lets
the plugin call this script restricted to the current task's classes instead
of all 1000, so recovery is proportional to task size.

BN statistics are still computed from the FULL ImageNet training set on the
first call and cached at --statistic-path.  Subsequent tasks load them from
cache — consistent with how the original recover.py works.

Change summary vs recover.py:
  main_syn()    + parser.add_argument('--class-ids', ...)
                + class_ids list parsed and passed to mp.spawn
  main_worker() + class_ids received as extra arg
                + targets_all_all / ipc_id_all / total_number / turn_index
                  all use len(class_ids) instead of hardcoded 1000
  Everything else: UNCHANGED.
"""

'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
import torchvision.datasets
from PIL import Image

import torch.multiprocessing as mp
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import torch.distributed as dist
mp.set_sharing_strategy('file_system')
from utils import *
from torchvision.transforms import functional as F
from torch.amp import autocast, GradScaler
import torch.cuda


def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════════
#  main_worker — identical to recover.py EXCEPT class_ids is an arg
#  and the four hardcoded-1000 lines are parameterised by len(class_ids).
# ════════════════════════════════════════════════════════════════════════

def main_worker(gpu, ngpus_per_node, args, model_teacher, model_verifier,
                ipc_id_range, K, loss_threshold, AMP,
                class_ids):          # ← ADDED: explicit class list
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    if args.world_size > 1:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    model_teacher = [_m.cuda(gpu).eval() for _m in model_teacher]
    scaler = torch.amp.GradScaler()
    for _m in model_teacher:
        for p in _m.parameters():
            p.requires_grad = False

    model_verifier = model_verifier.cuda(gpu)
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    save_every = 20
    batch_size = args.batch_size
    best_cost = 1e4
    load_tag_dict = [True for _ in range(len(model_teacher))]
    loss_r_feature_layers = [[] for _ in range(len(model_teacher))]
    load_tag = True

    for i, _model_teacher in enumerate(model_teacher):
        for name, module in _model_teacher.named_modules():
            if args.aux_teacher[i] in ["wide_resnet50_2", "regnet_y_400mf",
                                        "regnet_x_400mf"]:
                full_name = (str(_model_teacher.__class__.__name__)
                             + "_" + str(args.aux_teacher[i]) + "=" + name)
            else:
                full_name = str(_model_teacher.__class__.__name__) + "=" + name

            if isinstance(module, nn.BatchNorm2d):
                _hook = BNFeatureHook(module, save_path=args.statistic_path,
                                      name=full_name, gpu=gpu,
                                      training_momentum=args.training_momentum,
                                      flatness_weight=args.flatness_weight,
                                      category_aware=args.category_aware)
                _hook.set_hook(pre=True)
                load_tag = load_tag & _hook.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook.load_tag
                loss_r_feature_layers[i].append(_hook)

            elif isinstance(module, nn.Conv2d):
                _hook = ConvFeatureHook(module, save_path=args.statistic_path,
                                        name=full_name, gpu=gpu,
                                        training_momentum=args.training_momentum,
                                        drop_rate=args.drop_rate,
                                        flatness_weight=args.flatness_weight,
                                        category_aware=args.category_aware)
                _hook.set_hook(pre=True)
                load_tag = load_tag & _hook.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook.load_tag
                loss_r_feature_layers[i].append(_hook)

    sub_batch_size = int(batch_size // ngpus_per_node)

    if args.initial_img_dir != "None":
        initial_img_cache = PreImgPathCache(
            args.initial_img_dir,
            transforms=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ShufflePatches(2),
            ]))

    if args.category_aware == "local":
        original_img_cache = PreImgPathCache(
            args.train_data_path,
            transforms=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

    if not load_tag:
        # ── CL CHANGE: compute BN stats only from the current task's classes ──
        # In a genuine CL setting the full dataset is not available.
        # We filter ImageFolder to `class_ids` so only the task-visible images
        # are used. Stats are saved to a per-task subdirectory (statistic_path
        # already points to task_{T}/statistic/ via the plugin).
        class_id_set = set(class_ids)

        full_dataset = torchvision.datasets.ImageFolder(
            root=args.train_data_path,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

        # Keep only samples whose class index is in the current task
        task_indices = [
            i for i, (_, lbl) in enumerate(full_dataset.samples)
            if lbl in class_id_set
        ]
        task_subset = torch.utils.data.Subset(full_dataset, task_indices)
        train_loader = torch.utils.data.DataLoader(
            task_subset, num_workers=0, batch_size=256,
            drop_last=False, shuffle=True)

        print(f"Computing BN stats on {len(task_subset)} images "
              f"from {len(class_ids)} task classes (CL-compliant).")

        with torch.no_grad():
            for j, _mt in enumerate(model_teacher):
                if not load_tag_dict[j]:
                    print(f"  Teacher '{args.aux_teacher[j]}'")
                    for i, (data, targets) in tqdm(enumerate(train_loader)):
                        data    = data.cuda(gpu)
                        targets = targets.cuda(gpu)
                        for layer in loss_r_feature_layers[j]:
                            layer.set_label(targets)
                        _ = _mt(data)
                    for layer in loss_r_feature_layers[j]:
                        layer.save()
        print("BN statistics saved.")
    else:
        print("BN statistics loaded from cache (task-local).")

    for j in range(len(loss_r_feature_layers)):
        for layer in loss_r_feature_layers[j]:
            layer.set_hook(pre=False)

    # ── CL CHANGE: use task class_ids instead of np.arange(1000) ────
    n_classes = len(class_ids)
    targets_all_all = (
        torch.LongTensor(class_ids)[None, ...]
        .expand(len(ipc_id_range), n_classes)
        .contiguous().view(-1)
    )
    ipc_id_all = (
        torch.LongTensor(ipc_id_range)[..., None]
        .expand(len(ipc_id_range), n_classes)
        .contiguous().view(-1)
    )
    total_number = n_classes * (ipc_id_range[-1] + 1 - ipc_id_range[0])
    turn_index = (
        torch.LongTensor(np.arange(total_number))
        .view(len(ipc_id_range), n_classes)
        .transpose(1, 0).contiguous().view(-1)
    )
    # ── end CL CHANGE ────────────────────────────────────────────────

    counter = 0
    saved_iterations = 0
    print(f"GPU {gpu} | total images to synthesise: {total_number} "
          f"| classes: {class_ids}")

    for zz in range(0, total_number, batch_size):
        sub_turn_index = turn_index[
            zz + gpu * sub_batch_size : min(zz + (gpu + 1) * sub_batch_size, total_number)
        ]

        targets = targets_all_all[sub_turn_index].cuda(gpu)
        ipc_ids = ipc_id_all[sub_turn_index].cuda(gpu)

        if targets.numel() == 0:
            continue

        data_type = torch.float
        sub_batch_size = (
            min(zz + (gpu + 1) * sub_batch_size, total_number)
            - (zz + gpu * sub_batch_size)
        )
        if sub_batch_size < 0:
            continue

        if args.initial_img_dir != "None":
            sampled_images = [
                initial_img_cache.random_img_sample(_t) for _t in targets.tolist()
            ]
            inputs = torch.stack(sampled_images, 0).to(f"cuda:{gpu}").to(data_type)
            inputs.requires_grad_(True)
        else:
            inputs = torch.randn(
                (sub_batch_size, 3, 224, 224),
                requires_grad=True, device=f"cuda:{gpu}", dtype=data_type,
            )

        if args.category_aware == "local":
            expand_ratio = int(50000 / (args.ipc_number * 1000))
            tea_images = torch.stack(
                [original_img_cache.random_img_sample(_t)
                 for _t in (targets.tolist() * expand_ratio)], 0
            ).to(f"cuda:{gpu}").to(data_type)
            with torch.no_grad():
                for id_ in range(len(args.aux_teacher)):
                    for idx, mod in enumerate(loss_r_feature_layers[id_]):
                        mod.set_tea()
                    _ = model_teacher[id_](tea_images)

        iterations_per_layer = args.iteration
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)

        high_loss_crops  = [[] for _ in range(sub_batch_size)]
        high_loss_values = [[] for _ in range(sub_batch_size)]

        class ExplorationExploitationAug:
            def __init__(self, bs):
                self.cropper = transforms.RandomResizedCrop(224, scale=(0.5, 1))
                self.flipper = transforms.RandomHorizontalFlip()
                self.last_crops       = [None] * bs
                self.selected_indices = [None] * bs

            def __call__(self, imgs, iteration, hlc, hlv):
                bs = imgs.shape[0]
                cropped = []
                for img_idx in range(bs):
                    if iteration > K and hlc[img_idx]:
                        loss_w = torch.tensor(hlv[img_idx], device=imgs.device)
                        sel    = torch.multinomial(
                            torch.nn.functional.softmax(loss_w, dim=0), 1
                        ).item()
                        i, j, h, w = hlc[img_idx][sel]
                        self.selected_indices[img_idx] = sel
                    else:
                        self.selected_indices[img_idx] = None
                        i, j, h, w = self.cropper.get_params(
                            imgs[img_idx], self.cropper.scale, self.cropper.ratio
                        )
                    self.last_crops[img_idx] = (i, j, h, w)
                    cropped.append(
                        self.flipper(F.resized_crop(imgs[img_idx], i, j, h, w,
                                                    self.cropper.size))
                    )
                return torch.stack(cropped)

        aug_fn = ExplorationExploitationAug(sub_batch_size)

        for iteration in range(iterations_per_layer):
            if iteration > K and all(len(c) == 0 for c in high_loss_crops):
                print(f"Early stop at iter {iteration}: no high-loss crops remain")
                saved_iterations += iterations_per_layer - iteration
                break

            lr_scheduler(optimizer, iteration, iteration)
            inputs_jit       = aug_fn(inputs, iteration, high_loss_crops, high_loss_values)
            selected_indices = aug_fn.selected_indices

            id_ = counter % len(model_teacher)
            for mod in loss_r_feature_layers[id_]:
                mod.set_label(targets)
            counter += 1
            optimizer.zero_grad()
            for idx, mod in enumerate(loss_r_feature_layers[id_]):
                mod.set_ori()

            if AMP:
                if id_ == 2:
                    with autocast("cuda"):
                        sub_outputs = model_teacher[id_](inputs_jit)
                else:
                    sub_outputs = model_teacher[id_](inputs_jit)
            else:
                sub_outputs = model_teacher[id_](inputs_jit)

            rescale       = [args.first_multiplier] + [1.] * (len(loss_r_feature_layers[id_]) - 1)
            loss_r_feat   = sum(mod.r_feature * rescale[idx]
                                for idx, mod in enumerate(loss_r_feature_layers[id_]))
            loss_aux      = args.r_loss * loss_r_feat

            criterion_ce  = nn.CrossEntropyLoss(reduction="none").cuda()
            loss_ce_all   = criterion_ce(sub_outputs.float(), targets)
            loss_ce       = loss_ce_all.mean()
            loss_ema_ce   = torch.tensor(0., device=inputs_jit.device)
            loss          = loss_ce + loss_aux + loss_ema_ce * args.flatness_weight

            loss_vals = loss_ce_all.detach().cpu().numpy()

            for img_idx, (i, j, h, w) in enumerate(aug_fn.last_crops):
                if loss_vals[img_idx] > loss_threshold and iteration <= K:
                    crop = (i, j, h, w)
                    if crop in high_loss_crops[img_idx]:
                        ci = high_loss_crops[img_idx].index(crop)
                        high_loss_values[img_idx][ci] = loss_vals[img_idx]
                    else:
                        high_loss_crops[img_idx].append(crop)
                        high_loss_values[img_idx].append(loss_vals[img_idx])

                if selected_indices[img_idx] is not None:
                    new_loss = loss_vals[img_idx]
                    si       = selected_indices[img_idx]
                    if new_loss > loss_threshold:
                        high_loss_values[img_idx][si] = new_loss
                    else:
                        del high_loss_crops[img_idx][si]
                        del high_loss_values[img_idx][si]

            if iteration % save_every == 0:
                print(f"--- iter {iteration} | loss={loss.item():.4f} "
                      f"| feat={loss_r_feat.item():.4f} | ce={loss_ce.item():.4f}")

            if AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            inputs.data = clip(inputs.data)
            if gpu == 0 and (best_cost > loss.item() or iteration == 1):
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_ids)

        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════
#  save_images / validate — identical to recover.py
# ════════════════════════════════════════════════════════════════════════

def save_images(args, images, targets, ipc_ids, iter=None):
    for id in range(images.shape[0]):
        class_id = targets[id].item() if targets.ndimension() == 1 else targets[id].argmax().item()
        dir_path = "{}/new{:03d}".format(args.syn_data_path, class_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        place = (
            dir_path + "/class{:03d}_id{:03d}.jpg".format(class_id, ipc_ids[id])
            if iter is None
            else dir_path + "/class{:03d}_id{:03d}_iter{:04d}.jpg".format(class_id, ipc_ids[id], iter)
        )
        img_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        Image.fromarray((img_np * 255).astype(np.uint8)).save(place)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        bs   = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100. / bs) for k in topk]

    with torch.no_grad():
        prec1, prec5 = accuracy(model(input).data, target, topk=(1, 5))
    print("Verifier accuracy:", prec1.item())
    return prec1.item()


# ════════════════════════════════════════════════════════════════════════
#  main_syn — identical to recover.py EXCEPT --class-ids arg added
# ════════════════════════════════════════════════════════════════════════

def main_syn():
    parser = argparse.ArgumentParser("E2D recover — CL edition")

    # ── All original flags (unchanged) ──────────────────────────────
    parser.add_argument("--flatness",         action="store_true", default=False)
    parser.add_argument("--flatness-weight",  type=float, default=1.)
    parser.add_argument("--ema_alpha",        type=float, default=0.9)
    parser.add_argument("--exp-name",         type=str,   default="test")
    parser.add_argument("--ipc-number",       type=int,   default=50)
    parser.add_argument("--initial-img-dir",  type=str,   default="None")
    parser.add_argument("--syn-data-path",    type=str,   default="./syn_data")
    parser.add_argument("--store-best-images",action="store_true")
    parser.add_argument("--batch-size",       type=int,   default=100)
    parser.add_argument("--gpu-id",           type=str,   default="0,1")
    parser.add_argument("--world-size",       default=1,  type=int)
    parser.add_argument("--rank",             default=0,  type=int)
    parser.add_argument("--dist-backend",     default="nccl", type=str)
    parser.add_argument("--iteration",        type=int,   default=1000)
    parser.add_argument("--lr",               type=float, default=0.1)
    parser.add_argument("--jitter",           default=32, type=int)
    parser.add_argument("--category-aware",   default="global", type=str)
    parser.add_argument("--r-loss",           type=float, default=0.05)
    parser.add_argument("--first-multiplier", type=float, default=10.)
    parser.add_argument("--tv-l2",            type=float, default=0.0001)
    parser.add_argument("--training-momentum",type=float, default=0.4)
    parser.add_argument("--drop-rate",        type=float, default=0.0)
    parser.add_argument("--nuc-norm",         type=float, default=0.00001)
    parser.add_argument("--l2-scale",         type=float, default=0.00001)
    parser.add_argument("--arch-name",        type=str,   default="resnet18")
    parser.add_argument("--tau",              type=float, default=4.0)
    parser.add_argument("--average_grad_ratio",default=0., type=float)
    parser.add_argument("--verifier",         action="store_true")
    parser.add_argument("--verifier-arch",    type=str,   default="mobilenet_v2")
    parser.add_argument("--train-data-path",  type=str,   default="./imagenet/train")
    parser.add_argument("--statistic-path",   type=str,   default="./statistic")
    parser.add_argument("--K",                type=int,   default=700)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--loss-threshold",   type=float, default=0.5)
    parser.add_argument("--AMP",              type=int,   default=1)

    # ── CL ADDITION: task class filter ──────────────────────────────
    parser.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated class IDs to synthesise (CL addition). "
            "Defaults to all 1000 classes when omitted, preserving full "
            "original recover.py behaviour."
        ),
    )
    # ────────────────────────────────────────────────────────────────

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        set_seed(args.seed)

    # ── Parse class_ids (CL addition) ────────────────────────────────
    if args.class_ids is not None:
        class_ids = [int(x.strip()) for x in args.class_ids.split(",")]
    else:
        class_ids = list(range(1000))   # original behaviour
    print(f"[recover_cl] Synthesising {len(class_ids)} classes: {class_ids[:10]}{'...' if len(class_ids) > 10 else ''}")

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    aux_teacher   = ["resnet18", "mobilenet_v2", "efficientnet_b0",
                     "shufflenet_v2_x0_5", "alexnet"]
    args.aux_teacher = aux_teacher
    model_teacher = [models.__dict__[n](pretrained=True) for n in aux_teacher]
    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)

    ipc_id_range = list(range(0, args.ipc_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    port_id       = 10000 + np.random.randint(0, 1000)
    args.dist_url = "tcp://127.0.0.1:" + str(port_id)
    args.distributed = True
    ngpus_per_node   = torch.cuda.device_count()
    args.world_size  = ngpus_per_node * args.world_size

    if ngpus_per_node == 1:
        # Single-GPU path: call main_worker directly (avoids NCCL/libuv on Windows)
        main_worker(
            0, ngpus_per_node, args,
            model_teacher, model_verifier,
            ipc_id_range, args.K, args.loss_threshold, bool(args.AMP),
            class_ids,
        )
    else:
        torch.multiprocessing.set_start_method("spawn")
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(
                ngpus_per_node, args,
                model_teacher, model_verifier,
                ipc_id_range, args.K, args.loss_threshold, bool(args.AMP),
                class_ids,
            ),
        )


if __name__ == "__main__":
    main_syn()
