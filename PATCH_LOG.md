# Apple Silicon (MPS) 适配记录

## 背景

原始代码为 NVIDIA GPU / CUDA 12.8 环境编写，所有 device 均硬编码为 `'cuda'`。
本记录追踪为 M1 Max (32GB 统一内存, MPS 后端) 所做的所有改动，方便 debug 和回滚。

**目标机器：** Apple M1 Max · 32GB 统一内存  
**PyTorch 后端：** MPS（Metal Performance Shaders）  
**改动文件：** `common/utils/inference_utils.py`

---

## 改动清单

### PATCH-01 · 设备自动检测（新增）

**位置：** `inference_utils.py` 顶部 import 区块末尾（待确认行号）

**原始代码：** _(无，需新增)_

**修改后：**
```python
# Apple Silicon / CUDA / CPU 自动选择
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
```

**原因：** 统一所有 device 字符串的来源，后续改动只需替换变量。

---

### PATCH-02 · LayerDiff pipeline 设备迁移（共 5 处）

**位置：** `inference_utils.py` 第 65–69 行

**原始代码：**
```python
layerdiff_pipeline.vae.to(dtype=torch.bfloat16, device='cuda')
layerdiff_pipeline.trans_vae.to(dtype=torch.bfloat16, device='cuda')
layerdiff_pipeline.unet.to(dtype=torch.bfloat16, device='cuda')
layerdiff_pipeline.text_encoder.to(dtype=torch.bfloat16, device='cuda')
layerdiff_pipeline.text_encoder_2.to(dtype=torch.bfloat16, device='cuda')
```

**修改后：**
```python
layerdiff_pipeline.vae.to(dtype=torch.bfloat16, device=DEVICE)
layerdiff_pipeline.trans_vae.to(dtype=torch.bfloat16, device=DEVICE)
layerdiff_pipeline.unet.to(dtype=torch.bfloat16, device=DEVICE)
layerdiff_pipeline.text_encoder.to(dtype=torch.bfloat16, device=DEVICE)
layerdiff_pipeline.text_encoder_2.to(dtype=torch.bfloat16, device=DEVICE)
```

**原因：** `'cuda'` → `DEVICE` 变量，MPS 下自动使用 `'mps'`。  
**风险点：** `bfloat16` 在 MPS 上 PyTorch ≥ 2.4 已支持，若报错可降为 `float16`。

---

### PATCH-03 · LayerDiff group_offload（第 71 行）

**位置：** `inference_utils.py` 第 71 行

**原始代码：**
```python
layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
```

**修改后：**
```python
if DEVICE == 'cuda':
    layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
# MPS/CPU 不支持 enable_group_offload，跳过
# 若遇到 OOM，可尝试手动调低 resolution 参数
```

**原因：** `enable_group_offload` 是 CUDA 专用的显存卸载 API，MPS 上调用会直接报错。  
**影响：** 关掉后 MPS 不会启用分块卸载，但 32GB 统一内存足够容纳全部模型（约 14GB），不影响跑通。

---

### PATCH-04 · Marigold pipeline 设备迁移（第 195 行）

**位置：** `inference_utils.py` 第 195 行

**原始代码：**
```python
marigold_pipeline.to(device='cuda', dtype=torch.bfloat16)
```

**修改后：**
```python
marigold_pipeline.to(device=DEVICE, dtype=torch.bfloat16)
```

**原因：** 同 PATCH-02。

---

### PATCH-05 · Marigold group_offload（第 198 行）

**位置：** `inference_utils.py` 第 198 行

**原始代码：**
```python
marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
```

**修改后：**
```python
if DEVICE == 'cuda':
    marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
```

**原因：** 同 PATCH-03。

---

## 未改动项

| 行号 | 代码 | 说明 |
|------|------|------|
| 85 | `rng = torch.Generator(device=pipeline.unet.device)` | 从 pipeline 自身读取 device，PATCH-02 完成后自动正确，无需修改 |
| 273 | `depth_pred.to(device='cpu', ...)` | 固定转回 CPU 做 numpy 处理，逻辑正确，无需修改 |

---

## 已知风险 & Debug 备忘

| 风险 | 现象 | 处理方案 |
|------|------|----------|
| `bfloat16` MPS 不支持 | `RuntimeError: MPS does not support bfloat16` | PATCH-02/04 中改为 `float16` |
| MPS fallback 算子缺失 | `NotImplementedError: The operator ... is not implemented for MPS` | 加 `PYTORCH_ENABLE_MPS_FALLBACK=1` 环境变量，让该算子回退 CPU |
| 模型自动下载失败 | HuggingFace 网络超时 | 手动下载或设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `trans_vae` 属性不存在 | `AttributeError` | 说明该版本 LayerDiff 结构不同，检查 pipeline 实际属性 |

---

## 改动状态

| Patch | 状态 |
|-------|------|
| PATCH-01 设备检测 | ⬜ 待应用 |
| PATCH-02 LayerDiff 设备 | ⬜ 待应用 |
| PATCH-03 LayerDiff group_offload | ⬜ 待应用 |
| PATCH-04 Marigold 设备 | ⬜ 待应用 |
| PATCH-05 Marigold group_offload | ⬜ 待应用 |
