# To-Do List

## Issue: Multiple denoisers are performing worse than a single all-noise denoiser.
- Already tried changing the selection of low, medium, and high noise patches so as to send each patch to the proper network, to no avail.
  - Conclusion: The issue is not network selection, but rather, the high-noise and low-noise networks simply perform worse always (right now)
- Potential Solution: Use a different metric for noise estimation at train-time
  - Instead of training each specific-noise network on image patches with a **residual standard deviation** that lies within a particular range, we can instead use **PSNR** as our noise-level estimate. In this way, we train each specific-noise network on image patches with a **PSNR** that lies within a particular range.
  - **Hypothesis**: Residual standard deviation may not be a good unbiased estimator of noise level, because image patches with large but relatively uniform noise distribution would be categorized as "low-noise" because the standard deviation would be low. Instead, PSNR may be a better unbiased estimator of noise level, and is in fact the noise level estimate that we use for evaluation, anyways.
  - Currently training new models with the new PSNR-estimation scheme with the following PSNR ranges:
    - Low-noise model: 20.0 PSNR to 100.0 PSNR
    - Medium-noise model: 15.0 PSNR to 40.0 PSNR    
    - High-noise model: 0.00 PSNR to 30.0 PSNR    
