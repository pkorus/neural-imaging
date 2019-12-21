# Neural Imaging Toolbox 

Authors: [Paweł Korus](http://kt.agh.edu.pl/~korus/) and [Nasir Memon](http://isis.poly.edu/memon/), New York University

A Python toolbox for modeling and optimization of photo acquisition and distribution channels focusing on reliable manipulation detection capabilities. Joint optimization can include the camera ISP (NIP), lossy image compression (DCN), and forensic image analysis (FAN). 

The toolbox provides a [Tensorflow](https://www.tensorflow.org/) implementation of the following generic model: 

![training for optimized manipulation detection](docs/manipulation_detection_training_architecture.png)

More information can be found in papers listed below:

1. P. Korus, N. Memon, *Content Authentication for Neural Imaging Pipelines: End-to-end Optimization of Photo Provenance in Complex Distribution Channels*, CVPR'19, [arxiv:1812.01516](https://arxiv.org/abs/1812.01516) 
2. P. Korus, N. Memon, *Neural Imaging Pipelines - the Scourge or Hope of Forensics?*, 2019, [arXiv:1902.10707](https://arxiv.org/abs/1902.10707)
3. P. Korus, N. Memon, *Quantifying the Cost of Reliable Photo Authentication via High-Performance Learned Lossy Representations*, ICLR'20, [openreview](https://openreview.net/forum?id=HyxG3p4twS)

A standalone version of our lossy compression codec can be found in the [l3ic](https://github.com/pkorus/l3ic) repository.

## Change Log

- 2019.12 - Added support for learned compression, configurable manipulations + major refactoring

## Setup

The toolbox was written in Python 3. Follow the standard procedure to install dependencies.

```bash
> git clone https://github.com/pkorus/neural-imaging && cd neural-imaging
> pip3 install -r requirements.txt
> mkdir -p data/{raw,rgb}
> git submodule init
> cd pyfse && make && cd ..
```

#### Data Directory Structure

The toolbox uses the `data` directory to store images, training data and pre-trained models:

```
data/raw/                               - RAW images used for camera ISP training
  |- images/{camera name}                 RAW images (*.nef *.dng)
  |- nip_training_data/{camera name}      Bayer stacks (*.npy) and developed (*.png)
  |- nip_developed/{camera name}/{nip}    NIP-developed images (*.png)
data/rgb/                               - RGB images used for compression training
  |- kodak                                A sample dataset with kodak images
data/config                             - Training configuration files (e.g., DCN)
data/models                             - pre-trained TF models
  |- nip/{camera name}/{nip}              NIP models (TF checkpoints)
  |- dcn/{dcn model}                      DCN models (TF checkpoints)
data/m                                  - manipulation training results
data/results                            - CSV files with exported results
```

## Getting Started

The model can be easily customized to use various NIP models, photo manipulations, distribution channels, etc. Detailed configuration instruction are given below. We generally follow a 2-step protocol with separate model pre-training (camera ISP, compression) and joint optimization/fine-tuning for manipulation detection (retraining from scratch is also possible, but has not been tested). 

The following sections cover all training steps:

- [pre-training of the NIP models for faithful photo development](docs/pretrain_nip.md),
- [pre-training of the DCN for efficient lossy photo compression](docs/pretrain_dcn.md),
- [joint fine-tuning of the NIP/DCN/FAN models for reliable image manipulation detection](docs/optimization.md).

The optimization results are stored in JSON files (`training.json`) organized into a folder hierarchy (by default in `data/m/*`). The results can be [quickly visualized](docs/results.md) using the `results.py` script.

## Available Models

The framework provides:

- several [neural imaging pipelines](docs/pretrain_nip.md) (see below),
- a [differentiable JPEG codec](docs/optimization.md),
- a high-performance [neural image compression codec](docs/pretrain_dcn.md).

The toolbox currently provides the following imaging pipelines:

| Pipeline | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `libRAW` | uses the libRAW library to develop RAW images                |
| `Python` | simple Python implementation of a standard pipeline          |
| `INet`   | simple NIP which mimics step-by-step processing of the standard pipeline |
| `UNet`   | the well known UNet network                                  |
| `DNet`   | medium-sized model adapted from a recent architecture for joint demosaicing and denoising |
| `ONet`   | dummy pipeline for directly feeding RGB images into the workflow |

The standard pipelines are available in the `raw_api` module. Neural pipelines are available in `models/pipelines`. The `UNet` model was adapted from [Learning to See in the Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark).

Our neural image compression codec is an adaptation of the auto-encoder architecture proposed by Twitter (Theis et al., [Lossy Image Compression with Compressive Autoencoders](http://arxiv.org/abs/1703.00395)), and hence dubbed `TwitterDCN` (see `models/compression.py`). A standalone version is also available in the [neural-image-compression](https://github.com/pkorus/neural-image-compression) repository.

## Extending the Framework

The framework can be easily extended. See the [extensions](docs/extensions.md) section for information how to get started.

If you would like to contribute new models, training protocols, or if you find any bugs, please let us know. 

## Script Summary

| Script                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `develop_images.py`             | batch rendering of RAW images using various camera ISPs      |
| `diff_nip.py`                   | compare RAW rendering results of two camera ISPs             |
| `results.py`                    | visualization of FAN [optimization results](docs/results.md) |
| `test_dcn.py`                   | test neural image compression / generate rate-distortion profiles |
| `test_dcn_rate_dist.py`         | plot rate-distortion (R/D) curves for neural image compression (requires pre-computing R/D profiles using `test_dcn.py`) |
| `test_fan.py`                   | allows for testing trained FAN models on various datasets    |
| `test_framework.py`             | a rudimentary test of the entire framework (see [testing](docs/testing.md)) |
| `test_jpeg.py`                  | test differentiable approximation of the JPEG codec          |
| `test_nip.py`                   | test a pre-trained camera ISP                                |
| `train_dcn.py`                  | [pre-train lossy compression](docs/pretrain_dcn.md)          |
| `train_manipulation.py`         | optimization of the FAN (+NIP/DCN) for manipulation detection |
| `train_nip.py`                  | [pre-train camera ISPs](docs/pretrain_nip.md)                |
| `train_prepare_training_set.py` | prepare training data for camera ISP pre-training (imports RAW images) |
| `summarize_nip.py`              | extracts and summarizes performance stats for standalone NIP models |

## Data Sources

In our experiments we used RAW images from publicly available datasets: 

- MIT-5k - [https://data.csail.mit.edu/graphics/fivek/](https://data.csail.mit.edu/graphics/fivek/)
- RAISE - [http://loki.disi.unitn.it/RAISE/](http://loki.disi.unitn.it/RAISE/)

## Usage and Citations

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research in this direction. We have done our best to document, refactor, and test the code before publication. However, the toolbox is provided "as-is", without warranties of any kind.   

If you find this code useful in your work, please cite our papers:

```
@inproceedings{korus2019content,
  title={Content Authentication for Neural Imaging Pipelines: End-to-end Optimization of Photo Provenance in Complex Distribution Channels},
  author={Korus, Pawel and Memon, Nasir},
  booktitle={IEEE Conf. Computer Vision and Pattern Recognition},
  year={2019}
}
```
```
@article{korus2019neural,
  title={Neural Imaging Pipelines - the Scourge or Hope of Forensics?},
  author={Korus, Pawel and Memon, Nasir},
  journal={arXiv preprint arXiv:1902.10707},
  year={2019}
}
```
```
@inproceedings{korus2020quantifying,
  title={Quantifying the Cost of Reliable Photo Authentication via High-Performance Learned Lossy Representations},
  author={Korus, Pawel and Memon, Nasir},
  booktitle={IEEE Conf. Learning Representations},
  year={2020}
}
```

## Related Work

### End-to-end ISP optimization:

- Eli Schwartz, Raja Giryes, Alex M. Bronstein, [DeepISP: Towards Learning an End-to-End Image Processing Pipeline](https://arxiv.org/abs/1801.06724), 2019 - optimization for low-light performance
- Chen Chen, Qifeng Chen, Jia Xu, Vladlen Koltun, [Learning to See in the Dark](https://arxiv.org/abs/1805.01934), 2018 - optimization for low-light performance
- Marc Levoy, Yael Pritch [Night Sight: Seeing in the Dark on Pixel Phones](https://ai.googleblog.com/2018/11/night-sight-seeing-in-dark-on-pixel.html), 2018 - low-light optimization in Pixel 3 phones
- Steven Diamond, Vincent Sitzmann, Stephen Boyd, Gordon Wetzstein, Felix Heide, [Dirty Pixels: Optimizing Image Classification Architectures for Raw Sensor Data](https://arxiv.org/abs/1701.06487), 2017 - optimization for high-level vision
- Haomiao Jiang, Qiyuan Tian, Joyce Farrell, Brian Wandell, [Learning the Image Processing Pipeline](https://ieeexplore.ieee.org/document/7944641), 2017 - learning ISPs for non-standard CFA patterns
- Gabriel Eilertsen, Joel Kronander, Gyorgy Denes, Rafał K. Mantiuk, Jonas Unger, [HDR image reconstruction from a single exposure using deep CNNs](http://hdrv.org/hdrcnn/), 2017 - HDR simulation from a single exposure
- Felix Heide et al., [FlexISP: A Flexible Camera Image Processing Framework](http://www.cs.ubc.ca/labs/imager/tr/2014/FlexISP/), 2014 - general ISP optimization framework for various low-level vision problems

### Learned Compression

- Eirikur Agustsson, Michael Tschannen, Fabian Mentzer, Radu Timofte & Luc Van Gool, [Generative Adversarial Networks For Extreme Learned Image Compression](http://arxiv.org/abs/1804.02958), 2018 - using GANs to synthesize appearance on inconsequential content 
- Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte & Luc Van Gool, [Conditional Probability Models for Deep Image Compression](http://arxiv.org/abs/1801.04260), 2018 - adopts PixelCNN for context modeling
- Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston, [Variational Image Compression With A Scale Hyperprior](http://arxiv.org/abs/1802.01436), 2018 - an additional hyper-prior to handle spatial dependencies
- Lucas Theis, Wenzhe Shi, Andrew Cunningham & Ferenc Huszar, [Lossy Image Compression with Compressive Autoencoders](http://arxiv.org/abs/1703.00395), 2017 - deep auto-encoder competitive with JPEG2000
- Oren Rippel & Lubomir Bourdev, [Real-Time Adaptive Image Compression](http://arxiv.org/abs/1705.05823), 2017 - a high-performance lossy codec
- Johannes Ballé, Valero Laparra & Eero P. Simoncelli, [End-to-end Optimized Image Compression](http://arxiv.org/abs/1611.01704), 2016 - end-to-end optimization framework