# 💫CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching

Official repository for the paper "[CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching](https://arxiv.org/pdf/2404.03653.pdf)".

🌟 For more details, please refer to the project page: [https://caraj7.github.io/comat/](https://caraj7.github.io/comat/).

[[🌐 Webpage](https://caraj7.github.io/comat/)] [[📖 Paper](https://arxiv.org/pdf/2404.03653.pdf)] 




## 💥 News

- **[2024.09.26]** 🎉 CoMat is accepted by Neurips 2024!

- **[2024.04.30]** 🔥 We release the training code of CoMat.

- **[2024.04.05]** 🚀 We release our paper on [arXiv](https://arxiv.org/pdf/2404.03653.pdf).

  

## 👀 About CoMat

We propose 💫CoMat, an end-to-end diffusion model fine-tuning strategy with an image-to-text concept matching mechanism. We leverage an image captioning model to measure image-to-text alignment and guide the diffusion model to revisit ignored tokens.

![demo](fig/demo.png)

## 🔨Usage

### Install

Install the requirements first. We verify the environment setup in the current file but we expect the newer versions should also work.

```bash
pip install -r requirements.txt
```

The Attribute Concentration module requires [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to find the mask of the entities. Please run the following command to install Grounded-SAM.

```bash
mkdir seg_model
cd seg_model
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
mv Grounded-Segment-Anything gsam
cd gsam/GroundingDINO
pip install -e .
```



### Training

We currently support [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). Other Version 1 of Stable Diffusion should also be supported, e.g, [SD1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4), as they share the same architecture with SD1.5.

#### SD1.5

SD1.5 can be directly used to train.

First, we need to generate the latents used in the Fidelity Preservation module.

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/gan_gt_generate.py \
--prompt-path merged_data/abc5k_hrs10k_t2icompall_20k.txt \
--save-prompt-path train_data_sd15/gan_gt_data.jsonl \
--model-path runwayml/stable-diffusion-v1-5 \
--model-type sd_1_5
```

Then we start training.

```bash
bash scripts/sd15.sh
```

#### SDXL

We recommend to first fine-tune the Unet of SDXL on the resolution of 512\*512 to enable fast convergence  since the original SDXL generate images of poor quality on 512\*512. 

For the fine-tuning data, we directly use SDXL to generate 1024\*1024 images given the training prompt. Then we resize the generated images to 512\*512 and use these images to fine-tune SDXL for only 100 steps. We use the [script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py) from diffusers. We will later release the fine-tuned unet.

Then we first generate the latents with the fine-tuned UNet.

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/gan_gt_generate.py \
--prompt-path merged_data/abc5k_hrs10k_t2icompall_20k.txt \
--save-prompt-path train_data_sdxl/gan_gt_data.jsonl \
--unet-path FINETUNED_UNET_PATH \
--model-path stabilityai/stable-diffusion-xl-base-1.0 \
--model-type sdxl_unet
```

Finally we start training:

```bash
bash scripts/sdxl.sh
```



## 📌 TODO

- [ ] Release the checkpoints.

- [x] Release training code in April.

  

## :white_check_mark: Citation

If you find **CoMat** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{jiang2024comat,
  title={CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching},
  author={Jiang, Dongzhi and Song, Guanglu and Wu, Xiaoshi and Zhang, Renrui and Shen, Dazhong and Zong, Zhuofan and Liu, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2404.03653},
  year={2024}
}
```



## 👍Thanks

We would like to thank [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [TokenCompose](https://github.com/mlpc-ucsd/TokenCompose), and [Diffusers](https://github.com/huggingface/diffusers).
