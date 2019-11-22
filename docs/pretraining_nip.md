#  Pre-training the NIP (Camera ISP)

The camera ISP is replaced with a convolutional neural network (NIP) which replaces the following steps from the standard pipeline:

![neural imaging pipeline](nip_architectures_pipeline.png)

To pre-train a NIP model, we must first extract training data for a given camera. The training script looks for RAW images in `./data/raw/images/{camera name}`. By default, 150 horizontal images will be taken. This step produces pairs of RGGB Bayer stacks (stored in `*.npy` files) and RGB optimization targets (`*.png`).

```bash
> python3 train_prepare_training_set.py --cam EOS-4D
```

Then, we train selected NIP models (the `--nip` argument can be repeated). This step consumes (RGGB, RGB) training pairs and trains the NIP by optimizing the L2 loss on randomly sampled patches. By default, the 150 available images are split into 120/30 for training/validation.

```bash
> python3 train_nip.py --cam EOS-4D --nip INet --nip UNet
```

If needed, additional parameters for the NIPs can be provided as a JSON string.

```bash
> python3 train_nip.py --cam D7000 --nip INet --params '{"random_init": true}'
```

To validate the NIP models, you may wish to develop some images. The following command will develop all images in the data set. In this command, you can use all of the available imaging pipelines: `libRAW, Python, INet, DNet, UNet`.

```bash
> python3 develop_images.py {camera} {pipeline}
```

An example photograph developed with all of the available pipelines is shown below.  

![example NIP output](nip_output_example.jpg)

To quickly test a selected NIP on a central image patch (128 x 128 px by default):

```bash
> python3 test_nip.py --cam "Canon EOS 5D" --nip INet
```