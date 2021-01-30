# To-Do List

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
