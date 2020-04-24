## JPEG Codecs

The repository contains a differentiable model of JPEG compression which can be useful in other research as well (see `models.jpeg.DJPG`). The model expresses successive steps of the codec as  matrix multiplications or convolution layers (see papers for details) and supports the following approximations of DCT coefficient quantization:

- `None` - uses standard rounding (backpropagation not supported)
- `sin` - sinusoidal approximation of the rounding operator (allows for back-propagation)
- `soft` - uses standard rounding in the forward pass and sinusoidal approximation in the backward pass
- `harmonic` - a differentiable approximation with Taylor expansion 

See the test script `test_jpg.py` for a standalone usage example. The following plot compares image quality and generated outputs for various approximation modes.

![Differences between NIP models](dJPEG.png)
