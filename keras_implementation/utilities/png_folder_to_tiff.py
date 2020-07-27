"""
This file is used to convert a folder of PNG images to a single TIFF file
"""

import sys
from pathlib2 import Path
import argparse
import os


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--png_folder',
                    default='/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/Volume1/train/CoregisteredBlurryImages',
                    type=str,
                    help='The path to the folder containing PNG files')
parser.add_argument('--output_folder',
                    default='/home/ubuntu/PycharmProjects/MyDenoiser/keras_implementation/data/Volume1/train/CoregisteredBlurryImages',
                    type=str,
                    help='The path of the folder to which we save our .tiff')
parser.add_argument('--output_file_name',
                    default='output_tiff',
                    type=str,
                    help='The path of the folder to which we save our .tiff')
args = parser.parse_args()


# If no output_file_name is provided, give a default name
if args.output_file_name == '':
    args.output_file_name = Path(args.png_folder).parent.parent + Path(args.png_folder).parent


def main():
    """The main function of this file"""
    # Create the command to use the convert function from ImageMagick
    command = 'convert ' + \
              args.png_folder + '/' + \
              '*.png ' + \
              os.path.join(args.output_folder, args.output_file_name) + \
              '.tiff'

    # Execute the command
    os.system(command)

    print('Goodbye! Should be converted!')


if __name__ == "__main__":
    main()