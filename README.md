# Mini Transformer：从基础算子实现字符级语言模型

这是一个教学型 decoder-only Transformer。除参数容器、自动微分、优化器和常用激活/损失外，核心结构没有调用 PyTorch 的 `nn.Embedding`、`nn.Linear`、`nn.LayerNorm`、`nn.MultiheadAttention` 或融合注意力 API。

## 已实现组件

- `CharTokenizer`：字符排序建表、编码与解码
- `TokenEmbedding`：直接索引可学习参数矩阵
- `SinusoidalPositionalEncoding`：正弦/余弦位置编码
- `scaled_dot_product_attention`：手写 `softmax(QKᵀ/√d)V`
- `MultiHeadAttention`：Q/K/V 投影、拆头、合头和输出投影
- `FeedForward`：两层投影与 GELU
- `LayerNorm`：用均值、方差、缩放和平移参数实现
- `causal_mask`：下三角因果遮罩，禁止看到未来字符
- `TransformerBlock`：Pre-Norm、残差连接、注意力与 FFN
- 训练：随机序列批次、验证、梯度裁剪、最佳检查点和断点续训
- 生成：temperature 与 top-k 采样

## 项目结构

```text
mini_transformer.py  # 全部模型组件
train.py            # 训练入口
generate.py          # 生成入口
data/poems.txt        # 可直接使用的公版古诗小语料
tests/test_model.py  # 核心行为测试
```

## 安装

建议使用 Python 3.10+ 的虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

若有 NVIDIA 显卡，建议按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与 CUDA 对应的安装命令。

## 快速训练

CPU 冒烟训练（几十步，仅验证流程）：

```bash
python train.py --steps 20 --batch-size 8 --d-model 64 --heads 4 --layers 2 --ff-dim 128 --eval-interval 10 --eval-batches 2
```

默认小模型训练：

```bash
python train.py --data data/poems.txt --steps 2000
```

检查点默认保存在 `checkpoints/model.pt`。继续训练时，`--steps` 表示最终总步数：

```bash
python train.py --steps 4000 --resume
```

## 文本生成

提示文字中的每个字符必须在训练语料词表中：

```bash
python generate.py --prompt "春" --length 120 --temperature 0.8 --top-k 20
```

- temperature 越低越保守；越高越随机。
- top-k 只保留概率最高的 k 个候选；设为 `0` 表示不限制。
- 小型示例语料主要用于跑通流程。想获得更像样的文本，请换成更大且风格统一的 UTF-8 纯文本。

## 使用 Tiny Shakespeare

下载 `input.txt` 后直接指定路径；模型仍按字符（包含英文字符、空格和换行）训练：

```bash
python train.py --data input.txt --steps 5000 --seq-len 128
python generate.py --prompt "ROMEO:" --length 500
```

## 数据流

输入 `token_ids [B,T]` → 字符嵌入与位置编码 `[B,T,C]` → N 个 Transformer Block → LayerNorm → 与嵌入矩阵转置相乘 → logits `[B,T,V]`。训练目标是预测每个位置的下一个字符。

## 测试

```bash
pytest -q
```

测试覆盖 tokenizer 往返、因果遮罩、缩放点积注意力、LayerNorm，以及完整模型的前向和反向传播。

