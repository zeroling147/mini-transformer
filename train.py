"""Train the character-level Mini Transformer."""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import torch
from mini_transformer import CharTokenizer, MiniTransformer, ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="训练字符级 Mini Transformer")
    p.add_argument("--data", type=Path, default=Path("data/poems.txt"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def choose_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def get_batch(data, batch_size, seq_len, device):
    if len(data) <= seq_len:
        raise ValueError(f"数据仅有 {len(data)} 个字符，必须多于 seq_len={seq_len}")
    starts = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, datasets, args, device):
    model.eval()
    result = {}
    for name, data in datasets.items():
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(data, args.batch_size, args.seq_len, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def save_checkpoint(path, model, optimizer, tokenizer, step, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.config_dict(), "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "tokenizer": tokenizer.to_dict(),
                "step": step, "best_val": best_val}, path)


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    text = args.data.read_text(encoding="utf-8").lstrip("\ufeff")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cut = int(len(ids) * 0.9)
    datasets = {"train": ids[:cut], "val": ids[cut:]}
    if min(map(len, datasets.values())) <= args.seq_len:
        raise ValueError("训练集或验证集太短；请增加语料或减小 --seq-len")
    config = ModelConfig(tokenizer.vocab_size, args.d_model, args.heads, args.layers,
                         args.ff_dim, args.seq_len, args.dropout)
    model = MiniTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        if ckpt["tokenizer"] != tokenizer.to_dict():
            raise ValueError("当前训练文本的词表与检查点不一致")
        model = MiniTransformer(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step, best_val = ckpt["step"] + 1, ckpt.get("best_val", best_val)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device} | vocab={tokenizer.vocab_size} | params={params:,}")
    began = time.time()
    for step in range(start_step, args.steps):
        x, y = get_batch(datasets["train"], args.batch_size, args.seq_len, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps - 1:
            m = estimate_loss(model, datasets, args, device)
            print(f"step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f} | {time.time()-began:.1f}s")
            if m["val"] < best_val:
                best_val = m["val"]
                save_checkpoint(args.out, model, optimizer, tokenizer, step, best_val)
                print(f"  已保存最佳检查点: {args.out}")


if __name__ == "__main__": main()

