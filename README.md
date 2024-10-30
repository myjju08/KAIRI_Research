# TFG: Unified Training-Free Guidance for Diffusion Models

<p align="center">
Code and data for our paper <a href="#">TFG: Unified Training-Free Guidance for Diffusion Models</a>
    </br>
    </br>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9+-1f425f.svg?color=purple">
    </a>
    <a href="https://huggingface.co/docs/diffusers">
        <img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-0.26-blue">
    </a>
    <a>
        <img alt="MIT" src="https://img.shields.io/badge/License-MIT-yellow">
    </a>
</p>


- [Unified Training-free Guided Diffusion](#unified-training-free-guided-diffusion)
  - [Getting Started](#getting-started)


## 📰 News
* **[Jul. 12, 2024]**: We launch the first version of code for **TFG**. The current codebase supports tasks including label guidance, super resolution, gaussian deblur, fine-grained generation, audio declipping, combined guidance, style transfer, molecule property guidance. We will add more applications in the future.

## 👋 Overview

TBD

## 🚀 Set Up
1. **Install packages**. The important packages are in `requirements.txt`.
2. **Download resources**. Before each experiment, we need the *diffusion model*, the *guidance network* (maybe a classifier), and sometimes the *dataset* of this task (e.g., for super resolution).
   - You can download all the checkpoints from this [link](http://FS6712X-CIRCUS.ezconnect.to/share/BnQGlGGj6). Put them in `./models`.


## 💽 Usage
You can check `./scripts` for examplar scripts. Also, if you want to write the script yourself, refer to `./utils/configs.py` for details.

## 💫 Contributions
We would welcome any contributions, pull requests, or issues!
To do so, please either file a new pull request or issue. We'll be sure to follow up shortly!

## ✍️ Citation
If you find our work helpful, please use the following citations.
```
@article{
    ye2024tfg,
    title={TFG: Unified Training-Free Guidance for Diffusion Models},
    author={Haotian Ye and Haowei Lin and Jiaqi Han and Minkai Xu and Sheng Liu and Yitao Liang and Jianzhu Ma and James Zou and Stefano Ermon},
    booktitle={arxiv},
    year={2024}
}
```

## 🪪 License
MIT. Check `LICENSE.md`.