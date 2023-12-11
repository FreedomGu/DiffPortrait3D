

<p align="center">

  <h2 align="center">DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis</h2>
  <p align="center">
    <a href="https://boese0601.github.io/"><strong>Yuming Gu</strong></a><sup>1</sup>
    ·  
    <a href="https://seasonsh.github.io/"><strong>Hongyi Xu</strong></a><sup>2</sup>
    ·
    <a href="https://zerg-overmind.github.io/"><strong>You Xie</strong></a><sup>1</sup>
    ·
    <a href="https://www.linkedin.com/in/jessica-fu-60a504254/"><strong>Guoxian Song</strong></a><sup>1</sup>
    ·
    <a href="https://hongyixu37.github.io/homepage/"><strong>Yichun Shi</strong></a><sup>2</sup>
    ·
    <br><a href="https://guoxiansong.github.io/homepage/index.html"><strong>Di Chang</strong></a><sup>2</sup>
    ·  
    <a href="https://scholar.google.com/citations?user=0TIYjPAAAAAJ&hl=en"><strong>Jing Yang</strong></a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=_MAKSLkAAAAJ&hl=en"><strong>Lingjie Luo</strong></a><sup>2</sup>
    ·
    <a href="https://www.ihp-lab.org/"><strong>Mohammad Soleymani</strong></a><sup>1</sup>
    ·
    <br>
    <sup>1</sup>University of Southern California &nbsp;&nbsp;&nbsp; <sup>2</sup>ByteDance Inc.
    <br>
    </br>
        <a href="https://arxiv.org/abs/2311.12052">
        <img src='https://img.shields.io/badge/arXiv-MagicDance-green' alt='Paper PDF'>
        </a>
        <a href='https://boese0601.github.io/magicdance/'>
        <img src='https://img.shields.io/badge/Project_Page-MagicDance-blue' alt='Project Page'></a>
        <a href='https://youtu.be/VPJe6TyrT-Y'>
        <img src='https://img.shields.io/badge/YouTube-MagicDance-rgb(255, 0, 0)' alt='Youtube'></a>
  </p>
    </p>
<div align="center">
  <a href="https://youtu.be/VPJe6TyrT-Y"><img src="./figures/video_play.png" alt="MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer"></a>
</div>

*We propose MagicDance, a novel and effective approach to provide realistic human video generation enabling vivid motion and
facial expression transfer, and consistent 2D cartoon-style animation zero-shot generation without any fine-tuning. Thanks to MagicDance,
we can precisely generate appearance-consistent results, while the original T2I model (e.g., Stable Diffusion and ControlNet) can hardly
maintain the subject identity information accurately. Furthermore, our proposed modules can be treated as an extension/plug-in to the
original T2I model without modifying its pre-trained weight.*

<!-- *For avatar-centric video generation and animation, please also check our latest work <a href="">MagicAvatar</a>!* -->

 
## Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{chang2023magicdance,
      title={MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer}, 
      author={Di Chang and Yichun Shi and Quankai Gao and Jessica Fu and Hongyi Xu and Guoxian Song and Qing Yan and Xiao Yang and Mohammad Soleymani},
      year={2023},
      eprint={2311.12052},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public. We also appreciate the help from [Tan Wang](https://github.com/Wangt-CN), who offered assistance to our baselines comparison experiment.

* [DisCo](https://github.com/Wangt-CN/DisCo)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)
* [ControlNet](https://github.com/lllyasviel/ControlNet)

