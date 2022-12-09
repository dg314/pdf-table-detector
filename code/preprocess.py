from PIL import Image

import numpy as np
import random
import os

image_width, image_height = 354, 500

def extract_image(image_path):
    ori_image = Image.open(image_path)
    image_width_scale = image_width / ori_image.width
    image_height_scale = image_height / ori_image.height
    image = np.array(ori_image.resize((image_width, image_height)))
    return image, image_width_scale, image_height_scale

def extract_bounding_box(annotation_path, image_width_scale, image_height_scale):
    c0, r0, c1, r1 = float("inf"), float("inf"), -float("inf"), -float("inf")
    num_table_annotations = 0

    with open(annotation_path) as file:
        for line in file:
            annotations = line.split()

            if len(annotations) == 10 and annotations[9] == "table":
                num_table_annotations += 1

                c0 = min(c0, float(annotations[1]))
                r0 = min(r0, float(annotations[2]))
                c1 = max(c1, float(annotations[3]))
                r1 = max(r1, float(annotations[4]))

    return None if num_table_annotations == 0 else np.array([
        r0 * image_height_scale,
        c0 * image_width_scale,
        r1 * image_height_scale,
        c1 * image_width_scale,
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
        image_path = doc_root_name + "_ori.jpg"
        image, image_width_scale, image_height_scale = extract_image(image_path)

        annotation_path = doc_root_name + ".txt"
        bounding_box = extract_bounding_box(annotation_path, image_width_scale, image_height_scale)

        if bounding_box is None:
            continue

        insertion_index = random.randint(0, len(bounding_boxes))
        images.insert(insertion_index, image)
        bounding_boxes.insert(insertion_index, bounding_box)

    split_index = int(len(bounding_boxes) * train_test_split)

    train_images, train_bounding_boxes = np.array(images[:split_index]), np.array(bounding_boxes[:split_index])
    test_images, test_bounding_boxes = np.array(images[split_index:]), np.array(bounding_boxes[split_index:])
    
    return train_images, train_bounding_boxes, test_images, test_bounding_boxes