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


# This function was last used (and fully functional) as of Jan 6th, 2020, right before
# 3D retraining started.
def pair_data_generator_multiple_data_dirs(root_dirs,
                                           image_format=ImageFormat.PNG):
    """
    Provides a numpy array of training examples, given pahths to multiple training directories

    :param root_dirs: The paths of the training data directories
    :type root_dirs: list
    :param image_format: The format of image that our training data is (JPG or PNG)
    :type image_format: ImageFormat

    :return: training data
    :rtype: numpy.array
    """

    # initialize clear_data and clurry_data lists
    clear_data = []
    blurry_data = []

    # Iterate over the entire list of images
    for root_dir in root_dirs:
        print(f'Accessing training data in {root_dir}')
        for i, file_name in enumerate(os.listdir(join(root_dir, 'ClearImages'))):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                # Read the Clear and Blurry Images as numpy arrays
                clear_image = cv2.imread(os.path.join(root_dir, 'ClearImages', file_name), 0)
                blurry_image = cv2.imread(os.path.join(root_dir, 'CoregisteredBlurryImages', file_name), 0)

                # Histogram equalize the blurry image px distribution to match the clear image px distribution
                blurry_image = image_utils.hist_match(blurry_image, clear_image).astype('uint8')

                ''' Just logging 
                # Show the blurry image (Pre-Histogram Equalization), clear image, and
                # blurry image (Post-Histogram Equalization)
                logger.show_images([(f'CoregisteredBlurryImage (Pre-Histogram Equalization)', image),
                                    (f'Matching Clear Image', clear_image),
                                    ('CoregisteredBlurryImage (Post-Histogram Equalization)', equalized_image)])
                '''

                # Generate clear and blurry patches from the clear and blurry images, respectively...
                clear_patches, blurry_patches = generate_patch_pairs(clear_image=clear_image,
                                                                     blurry_image=blurry_image)

                # Append the patches to clear_data and blurry_data
                clear_data.append(clear_patches)
                blurry_data.append(blurry_patches)

    # Convert clear_data and blurry_data to numpy arrays of ints
    clear_data = np.array(clear_data, dtype='uint8')
    blurry_data = np.array(blurry_data, dtype='uint8')

    # Reshape clear_data, blurry_data, and std_data
    clear_data = clear_data.reshape((clear_data.shape[0] * clear_data.shape[1],
                                     clear_data.shape[2],
                                     clear_data.shape[3],
                                     1
                                     ))
    blurry_data = blurry_data.reshape((blurry_data.shape[0] * blurry_data.shape[1],
                                       blurry_data.shape[2],
                                       blurry_data.shape[3],
                                       1
                                       ))

    # Make sure that clear_data and blurry_data have the same length
    assert (len(blurry_data) == len(clear_data))

    ''' Commenting this out, as it will be done in train_datagen
    # Get the number of elements to discard
    num_elements_to_discard = len(clear_data) - len(clear_data) // batch_size * batch_size
    # Remove "num_elements_to_discard" elements from from clear_data and blurry_data
    clear_data = np.delete(clear_data, range(num_elements_to_discard), axis=0)
    blurry_data = np.delete(blurry_data, range(num_elements_to_discard), axis=0)
    '''

    return clear_data, blurry_data