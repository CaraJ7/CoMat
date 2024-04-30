# 💫CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching

Official repository for the paper "[CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching]()".

🌟 For more details, please refer to the project page: [https://caraj7.github.io/comat/](https://caraj7.github.io/comat/).

[[🌐 Webpage](https://caraj7.github.io/comat/)] [[📖 Paper]()] 


## 💥 News

- **[2024.04.30]** 🔥 We release the training code of CoMat.
- **[2024.04.05]** 🚀 We release our paper on [arXiv]().

## 👀 About CoMat

We propose 💫CoMat, an end-to-end diffusion model fine-tuning strategy with an image-to-text concept matching mechanism. We leverage an image captioning model to measure image-to-text alignment and guide the diffusion model to revisit ignored tokens.

![demo](/Users/cara/0Code/Opensource/CoMat/fig/demo.png)

## 🔨Usage

### Training

We current support SD1.5 and SDXL. Other Versions 1 of Stable Diffusion should also be supported, e.g, SD1.4.

## 📌 TODO

- [ ] Release the checkpoints.
- [x] Release training code in April.



## :white_check_mark: Citation

If you find **CoMat** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@inproceedings{jiang2024comat,
  title={CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching},
  author={Dongzhi Jiang, Guanglu Song, Xiaoshi Wu, Renrui Zhang, Dazhong Shen, Zhuofan Zong, Yu Liu, Hongsheng Li},
  booktitle={arXiv},
  year={2024}
}
```