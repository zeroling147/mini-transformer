"""Generate text from a trained checkpoint."""
import argparse
from pathlib import Path
import torch
from mini_transformer import CharTokenizer, MiniTransformer, ModelConfig
from train import choose_device


def main():
    p = argparse.ArgumentParser(description="使用 Mini Transformer 生成文本")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--prompt", default="春")
    p.add_argument("--length", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    model = MiniTransformer(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(prompt, args.length, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__": main()
