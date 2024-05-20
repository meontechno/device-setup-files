import os
import shutil
import random
import argparse
from tqdm import tqdm

import cv2


def copy_image(src_path, dest_path, size):
    if all(size):
        img = cv2.imread(src_path)
        cv2.imwrite(dest_path, cv2.resize(img, size))
    else:
        shutil.copy(src_path, dest_path)


def split_dataset(dataset_path, out_path, ratio=0.9, resize=[None, None], seed=None):
    print(f"Split ratio:\t{ratio}")
    random.seed(seed)

    train_path = os.path.join(out_path, "train")
    val_path = os.path.join(out_path, "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    num_classes = len(os.listdir(dataset_path))
    class_counter = 0

    for class_folder in os.listdir(dataset_path):
        class_counter += 1
        print(f"\nProcessing class {class_counter}/{num_classes}: {class_folder}")
        class_path = os.path.join(dataset_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        all_images = os.listdir(class_path)

        num_images = len(all_images)
        num_train = int(num_images * ratio)
        print(f"Found {num_images} images.")

        random.shuffle(all_images)

        train_images = all_images[:num_train]
        val_images = all_images[num_train:]

        for image in tqdm(train_images, desc="Train"):
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(train_path, class_folder, image)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            copy_image(src_path, dest_path, resize)

        for image in tqdm(val_images, desc="Val"):
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(val_path, class_folder, image)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            copy_image(src_path, dest_path, resize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--resize", nargs=2, type=int, default=[None, None])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    split_dataset(**vars(args))
