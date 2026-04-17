import os, sys, time, math, argparse, copy
import torch, torch.nn as nn
import numpy as np

#your own path to the model
sys.path.insert(0, ".")
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="/home/dashilovskiy/nanoGPT/checkpoints/lr0.001_nl6_nh8_bs32/ckpt.pt")
parser.add_argument("--data_dir",   default="/home/dashilovskiy/nanoGPT/nanoGPT/data/shakespeare_char")
parser.add_argument("--out_dir",    default="/home/dashilovskiy/nanoGPT/checkpoints/qat")
parser.add_argument("--qat_iters",  type=int,   default=1000)
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--block_size", type=int,   default=256)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--eval_iters", type=int,   default=200)
parser.add_argument("--log_interval", type=int, default=50)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(args.data_dir, "val.bin"),   dtype=np.uint16, mode="r")

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
    return x, y

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

def apply_dynamic_int8(model):
    return torch.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
    )

# ── 1. baseline ───────────────────────────────────────────────────────────────
print("="*60)
print("1. float32 baseline")
base = load_model(args.checkpoint)
base_loss, base_ppl = evaluate(base)
print(f"   loss={base_loss:.4f}  ppl={base_ppl:.4f}  size={size_mb(base):.1f}MB")

# ── 2. PTQ ────────────────────────────────────────────────────────────────────
print("\n2. PTQ int8 dynamic (no fine-tuning)")
ptq = apply_dynamic_int8(base)
ptq_loss, ptq_ppl = evaluate(ptq)
print(f"   loss={ptq_loss:.4f}  ppl={ptq_ppl:.4f}  size={size_mb(ptq):.1f}MB  delta={((ptq_ppl-base_ppl)/base_ppl*100):+.2f}%")

# ── 3. QAT fine-tuning with fake quant noise ──────────────────────────────────
print(f"\n3. QAT fine-tuning ({args.qat_iters} steps)")
print("   Strategy: simulate quantization noise during training,")
print("   then save float weights and apply dynamic int8\n")

qat = load_model(args.checkpoint)

# Manual fake-quant: add gaussian noise to weights each forward pass
# This is a simplified but effective QAT simulation without torch.quantization issues
class FakeQuantLinear(nn.Linear):
    def forward(self, x):
        if self.training:
            scale = self.weight.abs().max() / 127.0
            w_q = torch.round(self.weight / scale).clamp(-128, 127) * scale
            w_noisy = w_q
        else:
            w_noisy = self.weight
        return nn.functional.linear(x, w_noisy, self.bias)

def replace_linear_with_fakequant(model):
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Linear):
            fq = FakeQuantLinear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            fq.weight = mod.weight
            if mod.bias is not None:
                fq.bias = mod.bias
            setattr(model, name, fq)
        else:
            replace_linear_with_fakequant(mod)

replace_linear_with_fakequant(qat)
qat.train()

opt = torch.optim.AdamW(qat.parameters(), lr=args.lr)

def lr_fn(step):
    warmup = max(1, int(args.qat_iters * 0.1))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    p = (step - warmup) / max(1, args.qat_iters - warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * p))

t0 = time.time()
for step in range(args.qat_iters):
    for g in opt.param_groups:
        g["lr"] = lr_fn(step)
    x, y = get_batch("train")
    _, loss = qat(x, y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(qat.parameters(), 1.0)
    opt.step()
    if step % args.log_interval == 0 or step == args.qat_iters - 1:
        print(f"   step {step:4d}  loss={loss.item():.4f}  lr={lr_fn(step):.2e}  ({time.time()-t0:.0f}s)")

# ── 4. apply dynamic int8 to QAT-trained weights ─────────────────────────────
print("\n4. QAT weights + dynamic int8")
qat.eval()
qat_dyn = apply_dynamic_int8(qat)
qat_loss, qat_ppl = evaluate(qat_dyn)
print(f"   loss={qat_loss:.4f}  ppl={qat_ppl:.4f}  size={size_mb(qat_dyn):.1f}MB  delta={((qat_ppl-base_ppl)/base_ppl*100):+.2f}%")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Method':<25} {'Val loss':>10} {'Perplexity':>12} {'Size MB':>10} {'PPL delta':>12}")
print("-"*70)
rows = [
    ("float32_baseline",   base_loss, base_ppl, size_mb(base),    0.0),
    ("ptq_int8_dynamic",   ptq_loss,  ptq_ppl,  size_mb(ptq),     (ptq_ppl-base_ppl)/base_ppl*100),
    ("qat_int8_dynamic",   qat_loss,  qat_ppl,  size_mb(qat_dyn), (qat_ppl-base_ppl)/base_ppl*100),
]
for name, loss, ppl, sz, delta in rows:
    d = f"{delta:+.2f}%" if delta != 0 else "baseline"
    print(f"{name:<25} {loss:>10.4f} {ppl:>12.4f} {sz:>10.1f} {d:>12}")
print("="*70)

out = os.path.join(args.out_dir, "qat_results.txt")
with open(out, "w") as f:
    f.write(f"Checkpoint: {args.checkpoint}\nQAT iters: {args.qat_iters}\n\n")
    f.write(f"{'Method':<25} {'Val loss':>10} {'Perplexity':>12} {'Size MB':>10} {'PPL delta':>12}\n")
    f.write("-"*70 + "\n")
    for name, loss, ppl, sz, delta in rows:
        d = f"{delta:+.2f}%" if delta != 0 else "baseline"
        f.write(f"{name:<25} {loss:>10.4f} {ppl:>12.4f} {sz:>10.1f} {d:>12}\n")
print(f"Saved: {out}")
