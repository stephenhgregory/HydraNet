# To-Do List

## Issue (Major): Multiple denoisers are performing worse than a single all-noise denoiser.
- Already tried changing the selection of low, medium, and high noise patches so as to send each patch to the proper network, to no avail.
  - Conclusion: The issue is not network selection, but rather, the high-noise and low-noise networks simply perform worse always (right now)
- Potential Solution: Use a different metric for noise estimation at train-time
  - Instead of training each specific-noise network on image patches with a **residual standard deviation** that lies within a particular range, we can instead use **PSNR** as our noise-level estimate. In this way, we train each specific-noise network on image patches with a **PSNR** that lies within a particular range.
  - **Hypothesis**: Residual standard deviation may not be a good unbiased estimator of noise level, because image patches with large but relatively uniform noise distribution would be categorized as "low-noise" because the standard deviation would be low. Instead, PSNR may be a better unbiased estimator of noise level, and is in fact the noise level estimate that we use for evaluation, anyways.
  - Currently training new models with the new PSNR-estimation scheme with the following PSNR ranges:
    - Low-noise model: 20.0 PSNR to 100.0 PSNR
    - Medium-noise model: 15.0 PSNR to 40.0 PSNR    
    - High-noise model: 0.00 PSNR to 30.0 PSNR    

## Issue (Medium): Histogram Equalization creates issues with continuity when combining denoised slices
- We use both Histogram Equalization and Contrast-Limited Adaptive Histogram Equalization (CLAHE) to create more robustly denoised images.
- This makes the denoised output look great in discrete slices, but upon concatenation of the constituent slices, discontinuities appear between slices, rendering the denoised volume ugly and hard to deal with across other 2-dimensional planes (not sagittal)
- 2 Potential Solutions:
  1. Reverse Histogram Equalization:
    - We can simply perform the reverse of the separate histogram equalization functions, which will give us the desired continuity.
    - **However**, we then lose the fantastic benefit of histogram equalization.
  2. 3-dimensional Histogram Equalization as Post-Processing
    - We can add a new step to remove discontinuities by performing another Histogram Equalization across all slices in a volume.
    - This should fix the pixel intensity discontinuity problem.
    - How easy/hard is this?
    - Can we simply apply CLAHE to the entire volume? Or do we need to preserve an approximate intensity distribution from the input (noisy) volume?
    
## Issue (Medium): Slight checkerboard pattern in denoised output slices
- We denoise each slice by (currently 40x40) patches, then simply concatenate them together to obtain our output slice.
- Potential Solution:
  - Add a new denoiser network to HydraNet which takes as input the previous output of HydraNet, and produces a further denoised image.
  - This new network will function to equalize any approximate intensity differences among patches, remove checkerboard artifact, etc.
