# Extending the Framework

## Implementing New Pipelines

The toolbox pools available neural pipelines from the `models/pipelines` module. Implementation of new pipelines involves sub-classing `NIPModel` and providing implementations for the `construct_model` method and the `parameters` property. 

Network models are expected to use the provided input placeholder (`self.x`) and add attributes for model output (`self.y` and optionally `self.yy`). The standard output (`self.y`) should be clipped to [0,1]. For better learning stability, a non-clipped output can be provided (`self.yy`) - it will be automatically used for gradient computation. The models should use an optional string prefix (`self.label`) in variable names or named scopes. This facilitates the use of multiple NIPs in a single TF graph. 

## Implementing New Image Codecs

Lossy compression codecs are located in `models.compression` and should inherit from the `DCN` class which provides a general framework with quantization, entropy regularization, etc. already set up. Specific children classes should reimplement the `construct_model` method and use helper functions to set up quantization, e.g.:

```python
def construct_model(self, params):
  # Define hyper-parameters
  self._h = paramspec.ParamSpec({
    'n_features': (96, int, (4, 128)),
  })
  self._h.update(**params)
	# Encoder setup...
  latent = tf.contrib.layers.conv2d(self.x, ...)
  self.latent = self._setup_latent_space(latent)
  # Decoder setup...
  y = tf.contrib.layers.conv2d(self.latent, ...)
  y = tf.depth_to_space(y, ...)
  self.y = y
```

Use the `TwitterDCN` class as a reference.