#!/usr/bin/env python3
"""
Experiment A: QAT with more steps (1000 / 3000 / 5000)
Experiment B: QAT vs PTQ across model sizes (nl=6 vs nl=12)

Usage:
    python qat_ab.py --experiment a
    python qat_ab.py --experiment b
"""

import os, sys, time, math, argparse, copy
import torch, torch.nn as nn
import numpy as np


#your own path to the model
sys.path.insert(0, ".")
from model import GPT, GPTConfig

# ── args ──────────────────────────────────────────────────────────────────────

#path to data
parser = argparse.ArgumentParser()
parser.add_argument("--experiment",  choices=["a", "b"], required=True)
parser.add_argument("--data_dir",    default=".")
parser.add_argument("--ckpt_small",  default="./checkpoints/lr0.001_nl6_nh8_bs32/ckpt.pt")
parser.add_argument("--ckpt_large",  default="./checkpoints/lr0.001_nl12_nh8_bs32/ckpt.pt")
parser.add_argument("--out_dir",     default="./nanoGPT/checkpoints/qat_ab")
parser.add_argument("--batch_size",  type=int,   default=32)
parser.add_argument("--block_size",  type=int,   default=256)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--eval_iters",  type=int,   default=200)
parser.add_argument("--log_interval",type=int,   default=100)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────

train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(args.data_dir, "val.bin"),   dtype=np.uint16, mode="r")

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
    return x, y

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(path):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.float()

@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    avg = float(np.mean(losses))
    return avg, math.exp(avg)

def size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def apply_ptq(model):
    return torch.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
    )

# fake-quant linear layer
class FakeQuantLinear(nn.Linear):
    def forward(self, x):
        if self.training:
            scale = self.weight.abs().max() / 127.0
            w_q = torch.round(self.weight / scale).clamp(-128, 127) * scale
        else:
            w_q = self.weight
        return nn.functional.linear(x, w_q, self.bias)

def replace_with_fakequant(model):
    m = copy.deepcopy(model)
    def _replace(module):
        for name, mod in list(module.named_children()):
            if isinstance(mod, nn.Linear):
                fq = FakeQuantLinear(mod.in_features, mod.out_features,
                                     bias=mod.bias is not None)
                fq.weight = mod.weight
                if mod.bias is not None:
                    fq.bias = mod.bias
                setattr(module, name, fq)
            else:
                _replace(mod)
    _replace(m)
    return m

def run_qat(model, n_iters, lr=None):
    if lr is None:
        lr = args.lr
    qat = replace_with_fakequant(model)
    qat.train()
    opt = torch.optim.AdamW(qat.parameters(), lr=lr)

    def lr_fn(step):
        warmup = max(1, int(n_iters * 0.1))
        if step < warmup:
            return lr * (step + 1) / warmup
        p = (step - warmup) / max(1, n_iters - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    t0 = time.time()
    for step in range(n_iters):
        for g in opt.param_groups:
            g["lr"] = lr_fn(step)
        x, y = get_batch("train")
        _, loss = qat(x, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(qat.parameters(), 1.0)
        opt.step()
        if step % args.log_interval == 0 or step == n_iters - 1:
            print(f"      step {step:5d}/{n_iters}  loss={loss.item():.4f}"
                  f"  lr={lr_fn(step):.2e}  ({time.time()-t0:.0f}s)")

    qat.eval()
    return apply_ptq(qat)

def print_row(name, loss, ppl, base_ppl, sz, params=None):
    delta = f"{(ppl-base_ppl)/base_ppl*100:+.3f}%" if base_ppl else "baseline"
    p = f"  {params:.2f}M params" if params else ""
    print(f"   {name:<30} loss={loss:.4f}  ppl={ppl:.4f}"
          f"  size={sz:.1f}MB  delta={delta}{p}")

def save_results(filename, rows, header_extra=""):
    path = os.path.join(args.out_dir, filename)
    with open(path, "w") as f:
        if header_extra:
            f.write(header_extra + "\n\n")
        f.write(f"{'Method':<32} {'Val loss':>10} {'Perplexity':>12}"
                f" {'Size MB':>10} {'PPL delta':>12}\n")
        f.write("-"*80 + "\n")
        base_ppl = rows[0][2]
        for name, loss, ppl, sz in rows:
            d = f"{(ppl-base_ppl)/base_ppl*100:+.3f}%" if ppl != base_ppl else "baseline"
            f.write(f"{name:<32} {loss:>10.4f} {ppl:>12.4f} {sz:>10.1f} {d:>12}\n")
    print(f"\nSaved: {path}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — more QAT steps
# ═══════════════════════════════════════════════════════════════════════════════

if args.experiment == "a":
    STEPS_LIST = [0, 500, 1000, 3000, 5000]
    print("=" * 70)
    print("EXPERIMENT A: QAT quality vs number of fine-tuning steps")
    print(f"Checkpoint: {args.ckpt_small}")
    print(f"Steps: {STEPS_LIST}")
    print("=" * 70)

    base = load_model(args.ckpt_small)
    print(f"\nModel: {count_params(base):.2f}M params")

    base_loss, base_ppl = evaluate(base)
    print(f"\nBaseline float32: loss={base_loss:.4f}  ppl={base_ppl:.4f}")

    rows = [("float32_baseline", base_loss, base_ppl, size_mb(base))]
    all_results = []

    for n_steps in STEPS_LIST:
        print(f"\n{'─'*60}")
        if n_steps == 0:
            print(f"PTQ only (0 QAT steps):")
            model = apply_ptq(base)
        else:
            print(f"QAT ({n_steps} steps):")
            model = run_qat(base, n_steps)

        loss, ppl = evaluate(model)
        delta = (ppl - base_ppl) / base_ppl * 100
        print_row(f"qat_{n_steps}_steps", loss, ppl, base_ppl, size_mb(base))
        rows.append((f"qat_{n_steps}_steps", loss, ppl, size_mb(base)))
        all_results.append((n_steps, loss, ppl, delta))

    print("\n" + "=" * 80)
    print(f"{'Method':<32} {'Val loss':>10} {'Perplexity':>12} {'Size MB':>10} {'PPL delta':>12}")
    print("-" * 80)
    base_ppl_ref = rows[0][2]
    for name, loss, ppl, sz in rows:
        d = f"{(ppl-base_ppl_ref)/base_ppl_ref*100:+.3f}%" if ppl != base_ppl_ref else "baseline"
        print(f"{name:<32} {loss:>10.4f} {ppl:>12.4f} {sz:>10.1f} {d:>12}")
    print("=" * 80)

    # also print a simple table for plotting
    print("\nFor plotting (steps, ppl_delta%):")
    print(f"{'Steps':>8} {'PPL delta%':>12}")
    for n_steps, loss, ppl, delta in all_results:
        print(f"{n_steps:>8} {delta:>12.4f}")

    save_results("exp_a_results.txt", rows,
                 f"Experiment A: QAT steps vs quality\nCheckpoint: {args.ckpt_small}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — scaling: small vs large model
# ═══════════════════════════════════════════════════════════════════════════════

elif args.experiment == "b":
    QAT_ITERS = 1000

    print("=" * 70)
    print("EXPERIMENT B: Quantization impact — small (nl=6) vs large (nl=12)")
    print(f"QAT iters: {QAT_ITERS}")
    print("=" * 70)

    checkpoints = [
        ("nl6_nh8_bs32",  args.ckpt_small, "small (nl=6)"),
        ("nl12_nh8_bs32", args.ckpt_large, "large (nl=12)"),
    ]

    all_rows = []

    for tag, ckpt_path, label in checkpoints:
        print(f"\n{'='*60}")
        print(f"Model: {label}  |  {ckpt_path}")

        base = load_model(ckpt_path)
        n_params = count_params(base)
        print(f"Params: {n_params:.2f}M")

        # baseline
        base_loss, base_ppl = evaluate(base)
        print(f"  float32 baseline: loss={base_loss:.4f}  ppl={base_ppl:.4f}")

        # PTQ
        ptq = apply_ptq(base)
        ptq_loss, ptq_ppl = evaluate(ptq)
        ptq_delta = (ptq_ppl - base_ppl) / base_ppl * 100
        print(f"  PTQ int8:         loss={ptq_loss:.4f}  ppl={ptq_ppl:.4f}"
              f"  delta={ptq_delta:+.3f}%")

        # QAT
        print(f"  QAT ({QAT_ITERS} steps):")
        qat = run_qat(base, QAT_ITERS)
        qat_loss, qat_ppl = evaluate(qat)
        qat_delta = (qat_ppl - base_ppl) / base_ppl * 100
        print(f"  QAT int8:         loss={qat_loss:.4f}  ppl={qat_ppl:.4f}"
              f"  delta={qat_delta:+.3f}%")

        all_rows.append({
            "label":      label,
            "tag":        tag,
            "params":     n_params,
            "base_loss":  base_loss,
            "base_ppl":   base_ppl,
            "ptq_loss":   ptq_loss,
            "ptq_ppl":    ptq_ppl,
            "ptq_delta":  ptq_delta,
            "qat_loss":   qat_loss,
            "qat_ppl":    qat_ppl,
            "qat_delta":  qat_delta,
        })

    print("\n" + "=" * 80)
    print("SUMMARY — quantization sensitivity by model size")
    print("=" * 80)
    print(f"{'Model':<20} {'Method':<20} {'Val loss':>10} {'PPL':>10} {'PPL delta':>12}")
    print("-" * 80)
    for r in all_rows:
        print(f"{r['label']:<20} {'float32_baseline':<20} {r['base_loss']:>10.4f}"
              f" {r['base_ppl']:>10.4f} {'baseline':>12}")
        print(f"{r['label']:<20} {'ptq_int8':<20} {r['ptq_loss']:>10.4f}"
              f" {r['ptq_ppl']:>10.4f} {r['ptq_delta']:>11.3f}%")
        print(f"{r['label']:<20} {'qat_int8':<20} {r['qat_loss']:>10.4f}"
              f" {r['qat_ppl']:>10.4f} {r['qat_delta']:>11.3f}%")
        print()
    print("=" * 80)

    print("\nKey question: does larger model suffer MORE from quantization?")
    for r in all_rows:
        print(f"  {r['label']:<20} PTQ delta={r['ptq_delta']:+.3f}%"
              f"  QAT delta={r['qat_delta']:+.3f}%"
              f"  QAT improvement over PTQ: {r['ptq_delta']-r['qat_delta']:+.3f}%")

    out_path = os.path.join(args.out_dir, "exp_b_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Experiment B: quantization sensitivity by model size\n")
        f.write(f"QAT iters: {QAT_ITERS}\n\n")
        f.write(f"{'Model':<20} {'Method':<20} {'Val loss':>10} {'PPL':>10} {'PPL delta':>12}\n")
        f.write("-" * 80 + "\n")
        for r in all_rows:
            f.write(f"{r['label']:<20} {'float32_baseline':<20} {r['base_loss']:>10.4f}"
                    f" {r['base_ppl']:>10.4f} {'baseline':>12}\n")
            f.write(f"{r['label']:<20} {'ptq_int8':<20} {r['ptq_loss']:>10.4f}"
                    f" {r['ptq_ppl']:>10.4f} {r['ptq_delta']:>11.3f}%\n")
            f.write(f"{r['label']:<20} {'qat_int8':<20} {r['qat_loss']:>10.4f}"
                    f" {r['qat_ppl']:>10.4f} {r['qat_delta']:>11.3f}%\n\n")
    print(f"Saved: {out_path}")
