"""Contains old inference scripts which have been discardded. Storing them here in case I want to go gravedigging."""


def modified_main(args):
    """The main function of the program, modified  on Dec. 14, 2020"""

    # Get the latest epoch numbers
    latest_epoch_low_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_low_noise)
    latest_epoch_medium_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_medium_noise)
    latest_epoch_high_noise = model_functions.findLastCheckpoint(save_dir=args.model_dir_high_noise)

    # Create dictionaries to store models and training patches and load our 3 denoising models
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
