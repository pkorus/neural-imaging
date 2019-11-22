# Pre-training Lossy Compression

Our toolbox can support various DCN models, but by default we provide only one - `TwitterDCN`. This model follows the general autoencoder architecture by [Theis et al.](https://arxiv.org/abs/1703.00395) but uses custom solutions for quatization, entropy estimation and coding. We illustrate the structure of the model below.

![neural imaging pipeline](dcn-architecture.png)

We also provide 3 pre-trained versions of this model with low, medium and high-quality. The rate-distortion trade-off on 3 different datasets is shown below. 

![DCN rate-distortion trade-off](dcn_tradeoffs.png)

To train a different model, we can use the `train_dcn` script and provide a list of model configurations (csv file with hyper-parameter values) and an RGB training set, e.g.:

```bash
> python3 train_dcn.py --dcn TwitterDCN --split 31000:1000:1 --param_list data/config/twitter.csv --epochs 2500 --out data/models/dcn/custom --data data/rgb/compression/
```

We test the DCN using the `test_dcn` script which can:

- show compression results for a batch of images
- compare against JPEG compression at a matching quality level / bpp
- generate rate-distortion curves for various codecs

For example, to compare our low-quality codec against JPEG at matching bpp:

```bash
> python3 test_dcn.py --data ./data/rgb/kodak --dcn 16c jpeg-match-bpp --image 4
```

![neural imaging pipeline](dcn-example-low-kodak-4.jpg)