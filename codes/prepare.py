import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir_hr))):
        hr = pil_image.open(image_path).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr = convert_rgb_to_y(hr)

        for i in range(0, hr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, hr.shape[1] - args.patch_size + 1, args.stride):
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir_lr))):
        lr = pil_image.open(image_path).convert('RGB')
        lr = np.array(lr).astype(np.float32)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir_hr)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr = convert_rgb_to_y(hr)

        hr_group.create_dataset(str(i), data=hr)

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir_lr)))):
        lr = pil_image.open(image_path).convert('RGB')
        lr = np.array(lr).astype(np.float32)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir-hr', type=str, required=True)
    parser.add_argument('--images-dir-lr', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
