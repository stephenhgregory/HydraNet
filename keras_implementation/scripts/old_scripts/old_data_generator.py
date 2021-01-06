""" Contains old scripts for generating training data """

def pair_3d_data_generator(root_dir: str = join('data', 'Volume1', 'train'),
                           patch_size: Tuple[int, int, int] = (20, 20, 20), stride: int = 10,
                           scales: List[float] = [1, 0.9, 0.8, 0.7]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Provides a numpy array of training examples, given a path to a training directory

    Parameters
    ----------
    root_dir: The path of the training data directory
    patch_size: The size of each patches in pixels -> (patch_size, patch_size)
    stride: The stride with which to slide the patch-taking window
    scales: A list of scales at which we want to create image patches.

    Returns
    -------
    (clear_volume_patches, blurry_volume_patches)
    """
    # Obtain 3D image volumes
    clear_image_volume = image_utils.get_3d_image_volume(image_dir=join(root_dir, 'ClearImages'))
    blurry_image_volume = image_utils.get_3d_image_volume(image_dir=join(root_dir, 'CoregisteredBlurryImages'))

    # Histogram equalize the blurry volume px distribution to match the clear image px distribution
    blurry_image_volume = image_utils.hist_match(source=blurry_image_volume, template=clear_image_volume).astype(
        'uint8')

    # Generate clear and blurry patches from the clear and blurry images, respectively...
    clear_volume_patches, blurry_volume_patches = generate_3d_patch_pairs(clear_volume=clear_image_volume,
                                                                          blurry_volume=blurry_image_volume,
                                                                          patch_size=patch_size, stride=stride,
                                                                          scales=scales)

    # Convert clear_patches and blurry_patches to numpy arrays of ints
    clear_volume_patches = np.array(clear_volume_patches, dtype='uint8')
    blurry_volume_patches = np.array(blurry_volume_patches, dtype='uint8')

    # Make sure that clear_data and blurry_data have the same shape
    assert (len(clear_volume_patches) == len(blurry_volume_patches))

    return clear_volume_patches, blurry_volume_patches
