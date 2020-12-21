"""Contains old training functions"""

from deprecated import deprecated

@deprecated(reason="You should use my_train_datagen and my_train_datagen_single_model instead")
def my_new_train_datagen(epoch_iter=2000,
                         num_epochs=5,
                         batch_size=128,
                         data_dir=args.train_data,
                         noise_level=NoiseLevel.LOW,
                         low_noise_threshold=0.28,
                         high_noise_threshold=0.07):
    """
    Generator function that yields training data samples from a specified data directory.
    This function is replacing my_train_datagen, as that older function might be what's giving
    us lots of problems.

    :param epoch_iter: The number of iterations per epoch
    :param num_epochs: The total number of epochs
    :param batch_size: The number of training examples for each training iteration
    :param data_dir: The directory in which training examples are stored
    :param noise_level: The level of noise of the training data that we want
    :type noise_level: NoiseLevel
    :param low_noise_threshold: The lower residual image standard deviation threshold used to determine which data
                                should go to which network
    :type low_noise_threshold: float
    :param high_noise_threshold: The upper residual image standard deviation threshold used to determine which data
                                should go to which network
    :type high_noise_threshold: float

    :return: Yields a training example x and noisy image y
    """
    # Loop the following indefinitely...
    while True:
        # Set a counter variable
        counter = 0

        # If this is the first iteration...
        if counter == 0:
            print(f'Accessing training data in: {data_dir}')

            # Get training example from data_dir using data_generator
            x = data_generator.datagenerator(data_dir, image_type=data_generator.ImageType.CLEARIMAGE)
            y = data_generator.datagenerator(data_dir, image_type=data_generator.ImageType.BLURRYIMAGE)

            # Assert that the last iteration has a full batch size
            assert len(x) % args.batch_size == 0, \
                logger.log(
                    'make sure the last iteration has a full batchsize, '
                    'this is important if you use batch normalization!')
            assert len(y) % args.batch_size == 0, \
                logger.log(
                    'make sure the last iteration has a full batchsize, '
                    'this is important if you use batch normalization!')

            # Create lists to store all of the patches and stds for each noise level category
            x_low_noise = []
            y_low_noise = []
            stds_low_noise = []
            x_medium_noise = []
            y_medium_noise = []
            stds_medium_noise = []
            x_high_noise = []
            y_high_noise = []
            stds_high_noise = []
            x_all_noise = []
            y_all_noise = []
            stds_all_noise = []
            stds = []
            x_filtered = []
            y_filtered = []

            # Iterate over all of the image patches
            for x_patch, y_patch in zip(x, y):

                # If the patch is black (i.e. the max px value < 10), just skip this training example
                if np.max(x_patch) < 10:
                    continue

                # Get the residual std
                std = data_generator.get_residual_std(clear_patch=x_patch,
                                                      blurry_patch=y_patch)

                # Add the patches and their residual stds to their corresponding lists based on noise level
                x_all_noise.append(x_patch)
                y_all_noise.append(y_patch)
                stds_all_noise.append(x_patch)
                if std < low_noise_threshold:
                    x_low_noise.append(x_patch)
                    y_low_noise.append(y_patch)
                    stds_low_noise.append(std)
                    continue
                elif low_noise_threshold < std < high_noise_threshold:
                    x_medium_noise.append(x_patch)
                    y_medium_noise.append(y_patch)
                    stds_medium_noise.append(std)
                    continue
                elif std > high_noise_threshold:
                    x_high_noise.append(x_patch)
                    y_high_noise.append(y_patch)
                    stds_high_noise.append(std)
                    continue

            # Get x_filtered based upon the noise level that we're looking for
            if noise_level == NoiseLevel.LOW:
                x_filtered = x_low_noise
                y_filtered = y_low_noise
                stds = stds_low_noise
            elif noise_level == NoiseLevel.MEDIUM:
                x_filtered = x_medium_noise
                y_filtered = y_medium_noise
                stds = stds_medium_noise
            elif noise_level == NoiseLevel.HIGH:
                x_filtered = x_high_noise
                y_filtered = y_high_noise
                stds = stds_high_noise
            elif noise_level == NoiseLevel.ALL:
                x_filtered = x_all_noise
                y_filtered = y_all_noise
                stds = stds_high_noise

            # Convert image patches and stds into numpy arrays
            x_filtered = np.array(x_filtered, dtype='uint8')
            y_filtered = np.array(y_filtered, dtype='uint8')
            stds = np.array(stds, dtype='float64')

            # Remove elements from x_filtered and y_filtered so thatthey has the right number of patches
            discard_n = len(x_filtered) - len(y_filtered) // batch_size * batch_size;
            x_filtered = np.delete(x_filtered, range(discard_n), axis=0)
            y_filtered = np.delete(y_filtered, range(discard_n), axis=0)

            ''' Just logging '''
            # Plot the residual standard deviation
            image_utils.plot_standard_deviations(stds)

            # Standardize x and y to have a mean of 0 and standard deviation of 1
            # NOTE: x and y px values are centered at 0, meaning there are negative px values. We might have trouble
            # visualizing px that aren't either from [0, 255] or [0, 1], so just watch out for that
            x, x_orig_mean, x_orig_std = image_utils.standardize(x_filtered)
            y, y_orig_mean, y_orig_std = image_utils.standardize(y_filtered)

            ''' Just logging
            logger.print_numpy_statistics(x, "x (standardized)")
            logger.print_numpy_statistics(y, "y (standardized)")
            '''

            '''Just for logging
            # Save the reversed standardization of x and y into variables
            x_reversed = image_utils.reverse_standardize(x, x_orig_mean, x_orig_std)
            y_reversed = image_utils.reverse_standardize(y, y_orig_mean, y_orig_std)
            '''

            # Get a list of indices, from 0 to the total number of training examples
            indices = list(range(x.shape[0]))

            # Make sure that x and y have the same number of training examples
            assert indices == list(range(y.shape[0])), logger.log('Make sure x and y are paired up properly! That is, x'
                                                                  'is a ClearImage, and y is a CoregisteredBlurryImage'
                                                                  'but that the two frames match eachother. ')

            # Increment the counter
            counter += 1

        # Iterate over the number of epochs
        for _ in range(num_epochs):

            # Shuffle the indices of the training examples
            np.random.shuffle(indices)

            # Iterate over the entire training set, skipping "batch_size" at a time
            for i in range(0, len(indices), batch_size):
                # Get the batch_x (clear) and batch_y (blurry)
                batch_x = x[indices[i:i + batch_size]]
                batch_y = y[indices[i:i + batch_size]]

                '''Just logging
                # Get equivalently indexed batches from x_original, x_reversed, y_original, and y_reversed
                batch_x_original = x_original[indices[i:i + batch_size]]
                batch_x_reversed = x_reversed[indices[i:i + batch_size]]
                batch_y_original = y_original[indices[i:i + batch_size]]
                batch_y_reversed = y_reversed[indices[i:i + batch_size]]

                # Show some images from this batch
                logger.show_images(images=[("batch_x[0]", batch_x[0]),
                                         ("batch_x_original[0]", batch_x_original[0]),
                                         ("batch_x_reversed[0]", batch_x_reversed[0]),
                                         ("batch_y[0]", batch_y[0]),
                                         ("batch_y_original[0]", batch_y_original[0]),
                                         ("batch_y_reversed[0]", batch_y_reversed[0])])
                '''

                # Finally, yield x and y, as this function is a generator
                yield batch_y, batch_x