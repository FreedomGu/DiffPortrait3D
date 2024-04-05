

<p align="center">

  <h2 align="center">[<span style="color: red;">HighLight</span>][CVPR 2024]DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis </h2>
  <p align="center">
    <a href="https://www.yuming-gu.com/"><strong>Yuming Gu</strong></a><sup>1,2</sup>
    · 
    <a href="https://ge.in.tum.de/about/you-xie/"><strong>You Xie</strong></a><sup>2</sup>
    · 
    <a href="https://hongyixu37.github.io/homepage/"><strong>Hongyi Xu</strong></a><sup>2</sup>
    ·
    <a href="https://guoxiansong.github.io/homepage/index.html"><strong>Guoxian Song</strong></a><sup>2</sup>
    ·
    <a href="https://seasonsh.github.io/"><strong>Yichun Shi</strong></a><sup>2</sup>
    ·
    <br><a href="https://boese0601.github.io"><strong>Di Chang</strong></a><sup>1,2</sup>
    ·  
    <a href="https://jingyangcarl.com"><strong>Jing Yang</strong></a><sup>1</sup>
    ·
    <a href="http://linjieluo.com"><strong>Linjie Luo</strong></a><sup>2</sup>
    ·
    <br>
    <sup>1</sup>University of Southern California &nbsp;&nbsp;&nbsp; <sup>2</sup>ByteDance Inc.
    <br>
    </br>
        <a href="https://arxiv.org/abs/2312.13016">
        <img src='https://img.shields.io/badge/arXiv-DiffPortrait3D-green' alt='Paper PDF'>
        </a>
        <a href='https://freedomgu.github.io/DiffPortrait3D/'>
        <img src='https://img.shields.io/badge/Project_Page-DiffPortrait3D-blue' alt='Project Page'></a>
        <a href='https://www.youtube.com/watch?v=mI8RJ_f3Csw'>
        <img src='https://img.shields.io/badge/YouTube-DiffPortrait3D-rgb(255, 0, 0)' alt='Youtube'></a>
  </p>
    </p>
<div align="center">
  <a href="https://youtu.be/mI8RJ_f3Csw"><img src="./Figures/teaser.gif" alt="DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis"></a>
</div>


<!-- *We present DiffPortrait3D, a conditional diffusion model that is capable of synthesizing 3D-consistent photo-realistic novel views from as few as a single in-the-wild portrait. Specifically, given a single RGB input, we aim to synthesize plausible but consistent facial details rendered from novel camera views with retained both identity and facial expression. In lieu of time-consuming optimization and fine-tuning, our zero-shot method generalizes well to arbitrary face portraits with unposed camera views, extreme facial expressions, and diverse artistic depictions. At its core, we leverage the generative prior of 2D diffusion models pre-trained on large-scale image datasets as our rendering backbone, while the denoising is guided with disentangled attentive control of appearance and camera pose. To achieve this, we first inject the appearance context from the reference image into the self-attention layers of the frozen UNets. The rendering view is then manipulated with a novel conditional control module that interprets the camera pose by watching a condition image of a crossed subject from the same view. Furthermore, we insert a trainable cross-view attention module to enhance view consistency, which is further strengthened with a novel 3D-aware noise generation process during inference. We demonstrate state-of-the-art results both qualitatively and quantitatively on our challenging in-the-wild and multi-view benchmarks.* -->

<!-- *For avatar-centric video generation and animation, please also check our latest work <a href="">MagicAvatar</a>!* -->

## News
* **[2024.03.18]** Release code.
* **[2024.02.26]** <span style="color:red">**Congratulations to our team!**</span> Our paper has been accepted to <span > <b>CVPR2024</b></span>, see you in Seattle! 

* **[2023.12.28]** Release Diffportrait3D paper and project page.

## Getting Started

For pretrained checkpoint please download from google drive from [here](https://drive.google.com/file/d/14qzipHghFrs4CFpVo1xW9CZt8OGLJf9t/view?usp=drive_link).

Place the pretrained weights as following:

```bash
DiffPortrait3D
  |----checkpoints
    |----model_state-540000-001.th
```

## Environment
The environment from my machine is ```Python 3.8.5``` ```CUDA 11.7```. It's possible to have other compatible version.


```bash
conda env create -f diffportrait3D.yml
conda activate diffportrait3D
```

We test our code on NVIDIA V100, NVIDIA A100, NVIDIA A6000.
## Inference with small example_data:
```bash 
bash script/CVPR_Inference/inference_sample.sh
```
## Inference with your own in-the-wild data:

Due to company IP Policy, we cannot release the 3D aware noise model. In this case, we highly encourage you to acheive the 3D Aware Noise from other pretrained 3D GAN method. Models like [GOAE](https://github.com/jiangyzy/GOAE), [Triplanenet](https://github.com/anantarb/triplanenet) could also be a very good 3D-aware noise initial. Please also refer to [EG3D](https://nvlabs.github.io/eg3d/) to generate aligned camera condition.


## Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{gu2023diffportrait3d,
      title={DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis}, 
      author={Yuming Gu and Hongyi Xu and You Xie and Guoxian Song and Yichun Shi and Di Chang and Jing Yang and Lingjie Luo},
      year={2023},
      eprint={2312.13016},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Development

This research reference implementation is treated as a one-time code drop. Therefore, we may be slow in accepting external code contributions through pull requests.

## Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [EG3D](https://nvlabs.github.io/eg3d/)
* [PanoHead](https://github.com/SizheAn/PanoHead)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)
* [ControlNet](https://github.com/lllyasviel/ControlNet)

