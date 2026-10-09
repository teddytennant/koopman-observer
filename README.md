Implementation in JAX of the observation-corrected Koopman feature estimator (arXiv 2610.10366): https://arxiv.org/abs/2610.10366

```
pip install -e .
pytest
```

The package forms joint deep and shallow feature increments, fits a ridge EDMD affine operator and a Luenberger gain, and corrects predicted deep increments with the observed shallow innovation. Calibration is the ridge EDMD and Luenberger gain on synthetic feature increments. It does not match a frozen denoiser, ImageNet or CIFAR cohorts, or a latency table.
