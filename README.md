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

## 📰 News
* **[Oct. 30, 2024]**: We launch the first version of code for **TFG**. The current codebase supports tasks including label guidance, super resolution, gaussian deblur, fine-grained generation, audio declipping, combined guidance, style transfer, molecule property guidance. We will add more applications & models in the future. 

## 👋 Overview

Given an unconditional diffusion model and a predictor for a target property of interest (e.g., a classifier), the goal of training-free guidance is to generate samples with desirable target properties without additional training. Existing methods, though effective in various individual applications, often lack theoretical grounding and rigorous testing on extensive benchmarks. As a result, they could even fail on simple tasks, and applying them to a new problem becomes unavoidably difficult. This paper introduces a novel algorithmic framework encompassing existing methods as special cases, unifying the study of training-free guidance into the analysis of an algorithm-agnostic design space. Via theoretical and empirical investigation, we propose an efficient and effective hyper-parameter searching strategy that can be readily applied to any downstream task. We systematically benchmark across 7 diffusion models on 16 tasks with 40 targets, and improve performance by 8.5% on average. Our framework and benchmark offer a solid foundation for conditional generation in a training-free manner.


## 🚀 Set Up
1. **Install packages**. The important packages are in `requirements.txt`.
2. **Download resources**. Before each experiment, we need the *diffusion model*, the *guidance network* (maybe a classifier), and sometimes the *dataset* of this task (e.g., for super resolution).
   - You can download all the checkpoints from this [link](https://drive.google.com/drive/folders/1fS7dKpO4O-FjaLwuRXuHBxEOlkqMMTGh?usp=sharing). Set the `MODEL_PATH` in `utils/env_utils.py` as the path of the downloaded directory.

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
    booktitle={NeurIPS},
    year={2024}
}
```

## 🪪 License
MIT. Check `LICENSE.md`.
