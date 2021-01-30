"""Contains old inference scripts which have been discardded. Storing them here in case I want to go gravedigging."""


def modified_main(args):
    """The main function of the program, modified  on Dec. 14, 2020"""

    # Get the latest epoch numbers
    latest_epoch_low_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_low_noise)
    latest_epoch_medium_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_medium_noise)
    latest_epoch_high_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_high_noise)

    # Create dictionaries to store residual_std_models and training patches and load our 3 denoising residual_std_models
    model_dict = {
        "low": load_model(os.path.join(args.model_dir_low_noise, 'model_%03d.hdf5' % latest_epoch_low_noise),
                          compile=False),
        "medium": load_model(os.path.join(args.model_dir_medium_noise, 'model_%03d.hdf5' % latest_epoch_medium_noise),
                             compile=False),
        "high": load_model(os.path.join(args.model_dir_high_noise, 'model_%03d.hdf5' % latest_epoch_high_noise),
                           compile=False)
    }

    # If the result directory doesn't exist already, just create it
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # For each dataset that we wish to test on...
    for set_name in args.set_names:

        # If the <result directory>/<dataset name> doesn't exist already, just create it
        if not os.path.exists(os.path.join(args.result_dir, set_name)):
            os.mkdir(os.path.join(args.result_dir, set_name))

        # Get the upper and lower std thresholds (used to pick denoisers)
        low_std_thresh, upper_std_thresh = data_generator.get_lower_and_upper_percentile_stds(
            os.path.join(args.set_dir, set_name), lower_percentile=args.low_std_percentile,
            upper_percentile=args.upper_std_percentile)

        # Create a List of Peak Signal-To-Noise ratios (PSNRs) and Structural Similarities (SSIMs)
        psnrs = []
        ssims = []

        for image_name in os.listdir(os.path.join(args.set_dir, set_name, 'CoregisteredBlurryImages')):
            if image_name.endswith(".jpg") or image_name.endswith(".bmp") or image_name.endswith(".png"):

                # Get the image name minus the file extension
                image_name_no_extension, _ = os.path.splitext(image_name)

                # 1. Load the Clear Image x (as grayscale), and standardize the pixel values, and..
                # 2. Save the original mean and standard deviation of x
                x, x_orig_mean, x_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         str(set_name),
                                                                                         'ClearImages',
                                                                                         str(image_name)), 0))

                # Load the Coregistered Blurry Image y (as grayscale), and standardize the pixel values, and...
                # 2. Save the original mean and standard deviation of y
                y, y_orig_mean, y_orig_std = image_utils.standardize(imread(os.path.join(args.set_dir,
                                                                                         str(set_name),
                                                                                         'CoregisteredBlurryImages',
                                                                                         str(image_name)), 0))

                # Denoise the image
                start_time = time.time()
                x_pred = denoise_image_by_patches_no_cross_reference(y=y, file_name=str(image_name_no_extension),
                                                                     set_name=set_name,
                                                                     original_mean=x_orig_mean, original_std=x_orig_std,
                                                                     save_patches=False,
                                                                     single_denoiser=args.single_denoiser,
                                                                     model_dict=model_dict,
                                                                     low_std_thresh=low_std_thresh,
                                                                     upper_std_thresh=upper_std_thresh)
                print('%10s : %10s : %2.4f second' % (set_name, image_name, time.time() - start_time))

                ''' Just logging
                # Reverse the standardization
                x_pred_reversed = image_utils.reverse_standardize(x_pred,
                                                                  original_mean=x_orig_mean,
                                                                  original_std=x_orig_std)
                x_reversed = image_utils.reverse_standardize(x,
                                                             original_mean=x_orig_mean,
                                                             original_std=x_orig_std)
                y_reversed = image_utils.reverse_standardize(y,
                                                             original_mean=y_orig_mean,
                                                             original_std=y_orig_std)
                logger.show_images([("x", x),
                                    ("x_reversed", x_reversed),
                                    ("x_pred", x_pred),
                                    ("x_pred_reversed", x_pred_reversed),
                                    ("y", y),
                                    ("y_reversed", y_reversed)])
                '''

                # Reverse the standardization of x and x_pred (we actually don't need y at this point, only for logging)
                x = image_utils.reverse_standardize(x, original_mean=x_orig_mean, original_std=x_orig_std)
                x_pred = image_utils.reverse_standardize(x_pred, original_mean=x_orig_mean, original_std=x_orig_std)
                # y = image_utils.reverse_standardize(y, original_mean=y_orig_mean, original_std=y_orig_std)

                ''' Just logging
                logger.show_images([("x", x),
                                    ("x_pred", x_pred),
                                    ("y", y)])
                '''

                # Get the PSNR and SSIM for x
                psnr_x = peak_signal_noise_ratio(x, x_pred)
                ssim_x = structural_similarity(x, x_pred, multichannel=True)

                # If we want to save the result...
                if args.save_result:
                    ''' Just logging
                    # Show the images
                    logger.show_images([("y", y),
                                        ("x_pred", x_pred)])
                    '''

                    # Then save the denoised image
                    cv2.imwrite(filename=os.path.join(args.result_dir, set_name, image_name), img=x_pred)

                # Add the PSNR and SSIM to the lists of PSNRs and SSIMs, respectively
                psnrs.append(psnr_x)
                ssims.append(ssim_x)

        # Get the average PSNR and SSIM and add into their respective lists
        psnrs.append(np.mean(psnrs))
        ssims.append(np.mean(ssims))

        # If we want to save the result, save the result to <result_dir>/<set_name>/results.txt
        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_name, 'results.txt'))

        # Log the average PSNR and SSIM to the Terminal
        log('Dataset: {0:10s} \n  Average PSNR = {1:2.2f}dB, Average SSIM = {2:1.4f}'.format(set_name, np.mean(psnrs),
                                                                                             np.mean(ssims)))


def denoise_image_by_patches_no_cross_reference(y: np.ndarray, file_name: str, set_name: str, original_mean: float,
                                                original_std: float, low_std_thresh: float, upper_std_thresh: float,
                                                save_patches: bool = True, single_denoiser: bool = False,
                                                model_dict: Dict = None) -> np.ndarray:
    """
      Takes an input image and denoises it using a patch-based approach

      :param y: The input image to denoise
      :param file_name: The name of the file
      :param set_name: The name of the set containing our test data
      :param original_mean: The original mean px value of the image that the patch x is part of, which was used to
                              standardize the image
      :param original_std: The original standard deviation px valueof the image that the patch x is part of, which was
                              used to standardize the image
      :param low_std_thresh: The lower std threshold used to decide which denoiser to send each patch to
      :param upper_std_thresh: The upper std threshold used to decide which denoiser to send each patch to
      :param save_patches: True if we wish to save the individual patches
      :param single_denoiser: True if we wish to denoise patches using only a single denoiser
      :param model_dict: A dictionary of all the TF residual_std_models used to denoise image patches

      :return: x_pred: A denoised image as a numpy array
      :rtype: numpy array
      """

    # Set the save directory name
    save_dir_name = os.path.join(args.result_dir, set_name, file_name + '_patches')

    # First, create a denoised x_pred to INITIALLY be a deep copy of y. Then we will modify x_pred in place
    x_pred = copy.deepcopy(y)

    # Loop over the indices of y to get 40x40 patches from y
    for i in range(0, len(y[0]), 40):
        for j in range(0, len(y[1]), 40):

            # If the patch does not 'fit' within the dimensions of y, skip this and do not denoise
            if i + 40 > len(y[0]) or j + 40 > len(y[1]):
                continue

            # Get the (40, 40) patch, make a copy as a tensor, and then reshape the original to be a (40, 40, 1) patch
            y_patch = y[i:i + 40, j:j + 40]
            y_patch_tensor = image_utils.to_tensor(y_patch)
            y_patch = y_patch.reshape(y_patch.shape[0], y_patch.shape[1], 1)

            # Get the standard deviation of px values of y_patch
            y_patch_std = np.std(y_patch)

            # Make the denoiser assignment
            max_ssim_category = ''
            if y_patch_std < low_std_thresh:
                max_ssim_category = 'low'
            elif low_std_thresh < y_patch_std < upper_std_thresh:
                max_ssim_category = 'medium'
            elif y_patch_std > upper_std_thresh:
                max_ssim_category = 'high'

            # Inference with model_low_noise (Denoise y_patch_tensor to get x_patch_pred)
            x_patch_pred_tensor = model_dict[max_ssim_category].predict(y_patch_tensor)

            # Convert the denoised patch from a tensor to an image (numpy array)
            x_patch_pred = image_utils.from_tensor(x_patch_pred_tensor)

            # Replace the patch in x with the new denoised patch
            x_pred[i:i + 40, j:j + 40] = x_patch_pred

            if save_patches:
                # Reverse the standardization of x
                x_patch_pred = image_utils.reverse_standardize(x_patch_pred, original_mean, original_std)

                # Save the denoised patch
                image_utils.save_image(x=x_patch_pred,
                                       save_dir_name=save_dir_name,
                                       save_file_name=file_name + '_i-' + str(i) + '_j-' + str(j) + '.png')

    '''Just logging
    logger.show_images([("y", y), ("x_pred", x_pred)])
    '''

    return x_pred