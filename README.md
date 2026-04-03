
See-through: Single-image Layer Decomposition for Anime Characters
---

<a href='https://arxiv.org/abs/2602.03749'><img src='https://img.shields.io/badge/arXiv-2602.03749-b31b1b.svg'></a>
<a href='https://huggingface.co/spaces/24yearsold/see-through-demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Space-PSD%20Inference%20Demo-blue'></a>


_**[Jian Lin](https://github.com/dmMaze)<sup>1</sup>, [Chengze Li](https://moeka.me)<sup>1*</sup>, [Haoyun Qin](https://haoyunqin.com/)<sup>2,3,4</sup>, Kwun Wang Chan<sup>1</sup>, [Yanghua Jin](https://github.com/Aixile)<sup>3</sup>, [Hanyuan Liu](https://github.com/hyliu)<sup>1</sup>, Stephen Chun Wang Choy<sup>1</sup>, Xueting Liu<sup>1</sup>**_

<sup>1</sup>Saint Francis University &emsp; <sup>2</sup>University of Pennsylvania &emsp; <sup>3</sup>Spellbrush &emsp; <sup>4</sup>Shitagaki Lab

<sup>*</sup>Corresponding author

Conditionally accepted to appear in *ACM SIGGRAPH 2026 Conference Proceedings*.

</div>

---

## TL;DR

We introduce a framework that automates the transformation of static anime illustrations into manipulatable **2.5D models**. Our approach decomposes a single image into fully inpainted, semantically distinct layers with inferred drawing orders — up to **23 layers** including hair, face, eyes, clothing, accessories, and more.

![Our Representative Image](common/assets/representative.jpg)


<div align="center">
  

https://github.com/user-attachments/assets/023d271f-d8d7-4f6b-9083-96e714fb93e0


  <br>
  <em>This is our trailer video. Click to play.</em>
</div>

## Environment Setup

```bash
# 1. Create environment
conda create -n see_through python=3.12 -y
conda activate see_through

# 2. Install PyTorch (CUDA 12.8)
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies (includes common utilities and annotators)
pip install -r requirements.txt

# 4. Create assets symlink (you can also copy assets to the root if you prefer)
ln -sf common/assets assets
```

**Optional annotator tiers** (install as needed):

| Tier | Command | What it adds |
|------|---------|-------------|
| Body parsing | `pip install --no-build-isolation -r requirements-inference-annotators.txt` | detectron2 for body attribute tagging |
| SAM2 | `pip install --no-build-isolation -r requirements-inference-sam2.txt` | SAM2 for language-guided segmentation |
| Instance seg | `pip install -r requirements-inference-mmdet.txt` | mmcv/mmdet for anime instance segmentation |

> **Note:** Always run scripts from the repository root as the working directory.

## Scripts & Models

### Models

| Model | HuggingFace Repo | Description |
|-------|-----------------|-------------|
| LayerDiff 3D | <a href='https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d'><img src='https://img.shields.io/badge/%F0%9F%A4%97-layerdiff3d-blue'></a> | Diffusion-based transparent layer generation (SDXL) |
| Marigold Depth | <a href='https://huggingface.co/24yearsold/seethroughv0.0.1_marigold'><img src='https://img.shields.io/badge/%F0%9F%A4%97-marigold__depth-blue'></a> | Pseudo-depth estimation fine-tuned for anime |
| SAM Body Parsing | <a href='https://huggingface.co/24yearsold/l2d_sam_iter2'><img src='https://img.shields.io/badge/%F0%9F%A4%97-sam__body__parsing-blue'></a> | Semantic body part segmentation|

### Inference Scripts

| Script | Purpose |
|--------|---------|
| `inference/scripts/inference_psd.py` | **Main pipeline** — end-to-end layer decomposition → PSD output |
| `inference/scripts/syn_data.py` | Synthetic training data generation utilities |

> For the other inference/data parsing scripts refer to the [codebase](./inference/scripts/) and check the docstrings for details.

### Demo

| Notebook | Description |
|----------|-------------|
| `inference/demo/bodypartseg_sam.ipynb` | Interactive body part segmentation demo with visualization (19-parts) |

> For the definition of complete body tags, refer to [scrap_model.py](./common/live2d/scrap_model.py).

### Huggingface Space

We have prepared [a Huggingface Space](https://huggingface.co/spaces/24yearsold/see-through-demo) with ZeroGPU, so that if you register with HuggingFace, you should be able to run 1-2 PSD extractions per day (approximately 2-3 mins each, at 1280 resolution).

<img alt="image" src="https://github.com/user-attachments/assets/3f98f47b-e98b-4628-9859-8772cda69f93" />

(Copyright [Tohoku Zunko Project](https://zunko.jp/)).


## Usage

### Layer Decomposition (main pipeline)

`inference_psd.py` runs the full See-through pipeline: it applies the **LayerDiff 3D** model
for transparent layer generation and the fine-tuned **Marigold** model for pseudo-depth
inference, then stratifies the character into up to **23 semantic layers** and exports a
layered PSD file. Note that the separation for head and body are in two continuous stages, which
may lead to a longer time than the original model mentioned in the paper. 

```bash
# Decompose a single image into a layered PSD
python inference/scripts/inference_psd.py \
  --srcp assets/test_image.png \
  --save_to_psd

# Process a directory of images
python inference/scripts/inference_psd.py \
  --srcp path/to/image_folder/ \
  --save_to_psd
```

Output is saved to `workspace/layerdiff_output/` by default. Each result includes:
- A layered `.psd` file with semantically separated layers
- Intermediate depth maps and segmentation masks

> **Note:** This uses our most recent model with 23-layer body part separation (V3).


Once you have finished the layer splitting, you can further process the PSD with the scripts in `inference/scripts/heuristic_partseg.py` for depth-based or left-right stratification.


```bash
# Split based on depth
python inference/scripts/heuristic_partseg.py seg_wdepth --srcp workspace/test_samples_output/PV_0047_A0020.psd --target_tags handwear


#Left-right split
python inference/scripts/heuristic_partseg.py seg_wlr --srcp workspace/test_samples_output/PV_0047_A0020_wdepth.psd --target_tags handwear-1
```

### Low-VRAM Users

The default pipeline runs at bf16 precision and requires approximately 12-16 GB of VRAM at 1280 resolution.

**12 GB GPUs**: Enable group offload to reduce peak VRAM to ~10 GB at 1280 resolution:

```bash
python inference/scripts/inference_psd.py \
  --srcp assets/test_image.png \
  --save_to_psd \
  --group_offload
```

**8 GB GPUs**: Use the NF4 quantized pipeline, which uses 4-bit quantized model weights. This achieves ~8 GB peak VRAM at 1280 resolution, and can be further reduced by lowering the resolution with group offload:

```bash
# Install bitsandbytes (one-time)
pip install -r requirements-inference-bnb.txt

# Run with NF4 quantization (default: group_offload on, depth resolution 720)
python inference/scripts/inference_psd_quantized.py \
  --srcp assets/test_image.png \
  --save_to_psd

# For even lower VRAM, reduce layerdiff resolution to 1024
python inference/scripts/inference_psd_quantized.py \
  --srcp assets/test_image.png \
  --save_to_psd \
  --resolution 1024
```

The quantized models are hosted on HuggingFace and downloaded automatically on first run. Quality is close to the full-precision model (PSNR ~30 dB, SSIM ~0.96 vs bf16 baseline).

> **Note:** Group offload trades speed for VRAM savings (roughly 1.5x slower). NF4 quantization has minimal speed overhead but reduces model weight memory.

### Preparing the dataset for training (e.g., Live2D Parsing)

We have provided a separate repo for you to prepare the dataset for training the Live2D parsing model. Please refer to [CubismPartExtr](https://github.com/shitagaki-lab/CubismPartExtr) to know how to download the sample model files and prepare your workspace folder. 

After that, refer to the `README_datapipeline.md` for the instructions on how to run the data parsing scripts to prepare the dataset for inspection and training. 

### User Interface

Once you have prepared your data, you may go ahead with the user interfaces. Refer to [UI Readme](ui/README.md) for the instructions on how to launch the UI.

> We currently require the `workspace/datasets/` folder located at the repository root to launch the UI, as it contains the sample data for demonstration. We will work on making this more flexible in the future.

> We recommend installing the `mmdet` tier dependencies to ensure the UI can launch successfully. 


### Training

We have our training scripts ready, but we are still working on the documentation. We will release them no later than 2026/04/12. Please stay tuned!


## Community Support

We welcome community contributions and third-party integrations! 

If you build tools, extensions, or workflows on top of this project, please let us know by opening an issue or pull request — we would be happy to feature your work here.

- [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) by [@jtydhr88](https://github.com/jtydhr88) — Integration for ComfyUI, with node-based workflow and in-browser PSD export. Thank you for the amazing work!

We also seek i18n help for this project. Your help will be highly appreciated.


## Discussion: Is this Image-to-Live2D?

We don't think so — at least, not yet.

While we produce 2.5D layer decompositions from a single image,
the full Image-to-Live2D pipeline requires significantly more:

1. **Finer artistic decomposition.** Live2D models demand layers designed with specific
   deformation behaviors in mind. Our automatic decomposition prioritizes semantic
   correctness, but a Live2D artist would make different artistic choices about how
   to split layers for natural-looking motion.

2. **Rigging.** After decomposition, a Live2D model needs a deformation mesh, physics
   parameters, and motion curves — this rigging process is arguably the most critical
   (and labor-intensive) step, and it is not covered in this project.

3. **Artistic intent.** Professional Live2D works are crafted holistically: the layer
   structure, inpainting style, and rigging are designed together. Automating one step
   in isolation cannot replicate this.

That said, we believe our decomposition can serve as a useful **starting point** for
Live2D artists by eliminating some of the most tedious part of the workflow, such as manual segmentation
and occluded region inpainting.

## Changelog

**2026-04-02**
- Multiple memory optimizations; added suggestions for low-VRAM users (group offload, NF4 quantization).

## Acknowledgements

This work is funded and substantially supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project. No. UGC/FDS11/E02/23).

We would like to pay our thanks to the following people for their help and support:

+ [Dingkun Yan](https://scholar.google.com/citations?user=dM0hOpIAAAAJ&hl=en) and [Xinrui Wang](https://systemerrorwang.github.io/) for their inspiration and support on the project.
+ [USTC Student ACG Club "LEO"](https://space.bilibili.com/7021308) for kindly providing the sample Live2D model files for us to demonstrate on the paper.



This is an open-source research project.
We thank the authors of the following projects that made this work possible:

- [LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI) — Transparent image layer diffusion (Lvmin Zhang is always a legend)
- [Marigold](https://github.com/prs-eth/Marigold) — Diffusion-based monocular depth estimation
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Foundation model for segmentation
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — Open-set object detection
- [LaMa](https://github.com/advimman/lama) — Large mask inpainting
- [AnimeInstanceSegmentation](https://github.com/dreMaz/AnimeInstanceSegmentation) — Anime-specific instance segmentation

## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026seethrough,
  title={See-through: Single-image Layer Decomposition for Anime Characters},
  author={Lin, Jian and Li, Chengze and Qin, Haoyun and Chan, Kwun Wang and Jin, Yanghua and Liu, Hanyuan and Choy, Stephen Chun Wang and Liu, Xueting},
  journal={arXiv preprint arXiv:2602.03749},
  year={2026}
}
```
