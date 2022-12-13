from PIL import Image

import numpy as np
import random
import os

image_width, image_height = 256, 256
ann_width, ann_height = 1000, 1000

def extract_image(ori_image_path):
    ori_image = Image.open(ori_image_path)
    # image = np.array(ori_image.resize((image_width, image_height))) / 255.0
    image = np.array(ori_image.resize((image_width, image_height)))

    return image

def extract_bounding_box(annotation_path):
    c0, r0, c1, r1 = float("inf"), float("inf"), -float("inf"), -float("inf")

    with open(annotation_path) as file:
        num_table_annotations = 0

        for line in file:
            annotations = line.split()

            if len(annotations) == 10 and annotations[9] == "table":
                num_table_annotations += 1

                c0 = min(c0, float(annotations[1]))
                r0 = min(r0, float(annotations[2]))
                c1 = max(c1, float(annotations[3]))
                r1 = max(r1, float(annotations[4]))

        if num_table_annotations == 0:
            # No tables in image
            # return np.array([-1, -1, -1, -1])
            return None

        file.seek(0)

        for line in file:
            annotations = line.split()

            if len(annotations) == 10 and annotations[9] != "table":
                if float(annotations[1]) > c0 and float(annotations[2]) > r0 and float(annotations[3]) < c1 and float(annotations[4]) < r1:
                    # multi-table image
                    return None

    return np.array([
        int(r0 * image_height / ann_height),
        int(c0 * image_width / ann_width),
        int(r1 * image_height / ann_height),
        int(c1 * image_width / ann_width),
    ])

def get_data(data_dir="../data/docbank-samples", train_test_split=0.8):
    doc_root_names_half = set()
    doc_root_names_full = set()

    for file_name in os.listdir(data_dir):
        full_file_name = os.path.join(data_dir, file_name)

        if not os.path.isfile(full_file_name):
            continue

        root_name = None
        
        if full_file_name[-8:] == "_ori.jpg":
            root_name = full_file_name[:-8]
        elif file_name[-4:] == ".txt":
            root_name = full_file_name[:-4]

        if not root_name or root_name in doc_root_names_full:
            continue

        if root_name in doc_root_names_half:
            doc_root_names_half.remove(root_name)
            doc_root_names_full.add(root_name)
        else:
            doc_root_names_full.add(root_name)

    images, bounding_boxes = [], []

    for doc_root_name in doc_root_names_full:
        ori_image_path = doc_root_name + "_ori.jpg"

        annotation_path = doc_root_name + ".txt"
        bounding_box = extract_bounding_box(annotation_path)

        if bounding_box is None:
            continue

        image = extract_image(ori_image_path)

        insertion_index = random.randint(0, len(bounding_boxes))
        images.insert(insertion_index, image)
        bounding_boxes.insert(insertion_index, bounding_box)

    split_index = int(len(bounding_boxes) * train_test_split)

    train_images, train_bounding_boxes = np.array(images[:split_index]), np.array(bounding_boxes[:split_index])
    test_images, test_bounding_boxes = np.array(images[split_index:]), np.array(bounding_boxes[split_index:])
    
    return train_images, train_bounding_boxes, test_images, test_bounding_boxes