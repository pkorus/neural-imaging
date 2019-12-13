# Optimization for Manipulation Detection

The main script for training the entire workflow is `train_manipulation.py`. The script accepts command line parameters to control the model. An example command is shown and explained below:

```bash
> python3 train_manipulation.py \
--epochs=2501                `# Run optimization for 2,500 epochs` \
--end 3                      `# Repeat the experiment 3 times` \
--patch 128                  `# The output from NIP should be 128x128 RGB image` \
--ds none                    `# Skip down-sampling in the channel` \
--jpeg 50                    `# Use JPEG with QF 50 as channel compression` \
--nip DNet                   `# Use DNet model for camera ISP` \
--cam D90                    `# Use Nikon D90 images` \
--manip sharpen:1,gaussian,jpeg:80,awgn:4,median `# List of included manipulations` \
--dir ./data/m/jpeg-nip+/50  `# Output directory` \
--train nip                  `# Models for optimization, here only NIP` \
--ln 0.1 --ln 0.05 --ln 0.01 `# Repeat training for these regularization strenghts`
```

The list of included manipulations is comma-separated and can include optional strength (after a colon). The results will be generated into:

```bash
data/m/{experiment label}/{quality level}/{camera}/{nip}/{regularization}/{run number}/
# Example 1: Experiments with learned compression on native camera output
data/m/7-raw/dcn+/32c/D90/DNet/fixed-nip/lc-0.0500/000
# Example 2: Experiment with ISP optimization with JPEG-50 and UNet NIP
data/m/7-raw/jpeg-nip+/50/D90/UNet/ln-0.1000/000/
```

The script generates:
- `training.json` - JSON file with training progress and performance stats,
- `manip_validation_*.jpg` - visual presentation of training progress (change of loss, PSNR, etc. over time)
- `nip_validation_*.jpg` - current snapshot of patches developed by the NIP
- `models/{fan,...}` - current snapshot of the models (FAN and other optimized models)

## Forensics Analysis Network

Our Forensic Analysis Network (FAN) follows the state-of-the-art design principles and uses a constrained convolutional layer proposed in:

- Bayar, Belhassen, and Matthew C. Stamm. [Constrained convolutional neural networks: A new approach towards general purpose image manipulation detection.](https://ieeexplore.ieee.org/document/8335799) IEEE Transactions on Information Forensics and Security, 2018

While the original model used only the green channel, our FAN uses full RGB information for forensic analysis. See the `models.forensics.FAN` class for our Tensorflow implementation.

## JPEG Approximation

The repository contains a differentiable model of JPEG compression which can be useful in other research as well (see `models.jpeg.DJPG`). The model expresses successive steps of the codec as  matrix multiplications or convolution layers (see papers for details) and supports the following approximations of DCT coefficient quantization:

- `None` - uses standard rounding (backpropagation not supported)
- `sin` - sinusoidal approximation of the rounding operator (allows for back-propagation)
- `soft` - uses standard rounding in the forward pass and sinusoidal approximation in the backward pass
- `harmonic` - a differentiable approximation with Taylor expansion 

See the test script `test_jpg.py` for a standalone usage example. The following plot compares image quality and generated outputs for various approximation modes.

![Differences between NIP models](dJPEG.png)

## Batch Training

To simplify running experiments, the training script can be easily scripted (cf. `train_manipulation_batch.sh`).