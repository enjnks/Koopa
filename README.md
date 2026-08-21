# Koopa

This is the official codebase for the paper: [Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://arxiv.org/pdf/2305.18803.pdf), NeurIPS 2023. [[Slides]](https://cloud.tsinghua.edu.cn/f/407ef231c6cb4727a6fa/), [[Poster]](https://cloud.tsinghua.edu.cn/f/2f2fc7bd87d340ffaf29/).

## Updates

:triangular_flag_on_post: **News** (2024.2)  Introduction of our work in Chinese is available: [[Official]](https://mp.weixin.qq.com/s/10PoA6n51Qok-nJT6_vkhA), [[Zhihu]](https://www.zhihu.com/question/24189178/answer/3064876852).

:triangular_flag_on_post: **News** (2023.10) Koopa has been included in [[Time-Series-Library]](https://github.com/thuml/Time-Series-Library).

## Introduction

Koopa is a **lightweight**, **MLP-based**, and **theory-inspired** model for efficient time series forecasting. 

- Compared with the advanced but painstakingly trained deep forecasters, Koopa achieves state-of-the-art performance while saving **77.3%** training time and **76.0%** memory footprint.

<p align="center">
<img src="./figures/efficiency.png" height = "240" alt="" align=center />
</p>

- Focus on portraying ubiquitous **non-stationary** time series, Koopa shows **enhanced model capacity** empowered by the modern Koopman theory that naturally addresses the nonlinear evolution of real-world time series.
  
<p align="center">
<img src="./figures/motivation.png" height = "180" alt="" align=center />
</p>

- Koopa differs from the canonical Koopman Autoencoder without the reconstruction loss function to achieve **end-to-end predictive training**.
  
<p align="center">
<img src="./figures/architecture.png" height = "360" alt="" align=center />
</p>


## Discussions

There are already several discussions about our paper, we appreciate a lot for their valuable comments and efforts: [[Official]](https://mp.weixin.qq.com/s/10PoA6n51Qok-nJT6_vkhA), [[Openreview]](https://openreview.net/forum?id=jsanMaAxZE), [[Zhihu]](https://www.zhihu.com/question/24189178/answer/3064876852).


## Preparation

1. Install Pytorch (>=1.12.0) and other necessary dependencies.
```
pip install -r requirements.txt
```
2. All the six benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).

## Training scripts

We provide the Koopa experiment scripts and hyperparameters of all benchmark datasets under the folder `./scripts`.

```bash
bash ./scripts/ECL_script/Koopa.sh
bash ./scripts/Traffic_script/Koopa.sh
bash ./scripts/Weather_script/Koopa.sh
bash ./scripts/ILI_script/Koopa.sh
bash ./scripts/Exchange_script/Koopa.sh
bash ./scripts/ETT_script/Koopa.sh
```

## Applicable for Rolling Forecast

- By adapting the operator on the incoming time series during rolling forecast, the proposed model can achieve more accurate performance via adapting to continuous distribution shift.

- The naïve implementation of operator adaptation is based on the DMD algorithm. We propose an iterative algorithm with reduced complexity. The details can be found in the Appendix of our paper.


- We also provide a tutorial notebook for a better understanding of this scenario. See `operator_adaptation.ipynb` for the details.
<p align="center">
<img src="./figures/algorithm.png" height = "480" alt="" align=center />
</p>

## Threshold visualization

`figures/stsa_threshold_1192.png` is the corrected transparent image.
`figures/stsa_threshold_1192.svg` is a self-contained SVG version that embeds
the corrected pixels without changing the chart dimensions.

- STSA: 1140 (green dashed line)
- μ+3σ: 1192 (orange dashed line)

The source chart is 1359 × 1000 pixels. Its zero point is at x=290 and the
STSA marker center is at x=1021. Using the same linear scale gives the new
μ+3σ center:

```text
290 + (1021 - 290) × 1192 / 1140 ≈ 1054
```

Only the old marker strip and the new marker strip are edited. Transparency,
axes, labels, legend, and all other source pixels are retained. The reusable
utility is `utils/move_threshold_marker.py`.

The legend region is converted from incorrectly premultiplied RGBA values to
standard straight-alpha PNG values. This keeps its background white when the
image is inserted into Word instead of appearing gray.

To avoid muted colors in Word, chart lines use the same standard RGB values as
the original plot:

- Feature blue: `RGB(0, 0, 255)` / `#0000FF`
- μ+3σ orange: `RGB(255, 165, 0)` / `#FFA500`
- STSA green: `RGB(0, 128, 0)` / `#008000`

Only RGB channels are normalized; alpha values and pixel coordinates are not
changed, so line positions, widths, antialiasing, and dash patterns stay fixed.

## PHM2012 TSP sensitivity experiment

The experiment requirements transcribed from the reference image, including
formulas, result tables, and two Markdown/Mermaid chart examples, are available
in [`docs/phm2012_tsp_sensitivity_experiment.md`](docs/phm2012_tsp_sensitivity_experiment.md).

## Chart without the μ+3σ marker

`figures/image3_without_mu3sigma.png` removes the orange/yellow μ+3σ vertical
marker and its legend row while retaining the Feature curve, STSA marker,
axes, labels, original dimensions, and transparency. A self-contained SVG is
available at `figures/image3_without_mu3sigma.svg`.

The reusable pixel-preserving utility is `utils/remove_mu3sigma.py`. It also
compacts the legend to two rows and restores the axis pixels previously covered
by the removed marker. Because the μ+3σ marker overlapped the Feature signal,
the hidden blue segment is reconnected from the neighboring trajectory so the
result remains continuous after marker removal.


## Citation

If you find this repo useful, please cite our paper. 

```
@article{liu2023koopa,
  title={Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors},
  author={Liu, Yong and Li, Chenyu and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2305.18803},
  year={2023}
}
```

## Contact

If you have any questions or want to use the code, please contact:
* liuyong21@mails.tsinghua.edu.cn
* lichenyu20@mails.tsinghua.edu.cn
