"""
Script to uscale images using superresolution AI.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Silence tensorflow messages.

import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from ISR.models import RDN

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
    ".tif", ".TIF", ".tiff", ".TIFF",
]

def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_images(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def upscale_image_file(model, image_path, image_save_path):
    img = np.array(Image.open(image_path))
    sr_img = model.predict(img)
    Image.fromarray(sr_img).save(image_save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model-type", type=str, default="DE_L_RDN", help="Model type (L_RDN, S_RDN or DE_L_RDN)")
    parser.add_argument("-m", "--model-filepath", type=str, default="weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to image or folder where input images are stored.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to image or folder where upscaled images will be stored.")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Overwrite file at destination.")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = find_images(args.input)
    else:
        print(f"Error: '{args.input}' does not exist ...")
        exit(1)

    # If output is a directory then we have to make sure that it exists and is not a file.
    if not is_image(args.output):
        if os.path.isfile(args.output):
            if args.overwrite:
                os.remove(args.output)
            else:
                print(f"Error: '{args.output}' already exists!")
                exit(1)

        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # Create target file paths.
    if is_image(args.output):
        image_save_paths = [args.output]
    else:
        image_save_paths = []
        for path in image_paths:
            image_save_paths.append(os.path.join(args.output, os.path.basename(path)))

    # If we do not overwrite we need to make sure that files do not already exists.
    if not args.overwrite:
        for path in image_save_paths:
            if os.path.isfile(path):
                print(f"Error: file '{path}' already exists at destination!")
                exit(1)

    # Load model
    if args.model_type == "L_RDN":
        model = RDN(arch_params={'C': 6, 'D':20, 'G':64, 'G0':64, 'x':2})
    elif args.model_type == "S_RDN":
        model = RDN(arch_params={'C': 3, 'D':10, 'G':64, 'G0':64, 'x':2})
    elif args.model_type == "DE_L_RDN":
        model = RDN(arch_params={'C': 6, 'D':20, 'G':64, 'G0':64, 'x':2})
    else:
        print(f"Error: Model type '{args.model_type}' is not supported! Only L_RDN, S_RDN or DE_L_RDN are supported!")
        exit(1)

    if len(image_paths) == 0:
        print(f"Error: no images in '{args.input}' !")
        exit(1)

    model.model.load_weights(args.model_filepath)

    if len(image_paths) == 1:
        print(f"Processing' {image_paths[0]}'... ")
        upscale_image_file(model, image_paths[0], image_save_paths[0])
    else:
        i = 0
        for image_path, image_save_path in zip(image_paths, image_save_paths):
            i += 1
            print(f"[{i}/{len(image_paths)}] Processing '{image_path}'....]")
            upscale_image_file(model, image_path, image_save_path)

if __name__ == "__main__":
    main()
