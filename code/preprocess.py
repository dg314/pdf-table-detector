from PIL import Image

import numpy as np
import random
import os

def extract_image(image_path):
    image = Image.open(image_path)
    image = image.resize((354, 500))
    return np.array(image)

def extract_bounding_box(annotation_path):
    x0, y0, x1, y1 = float("inf"), float("inf"), -float("inf"), -float("inf")
    num_table_annotations = 0

    with open(annotation_path) as file:
        for line in file:
            annotations = line.split()

            if len(annotations) == 10 and annotations[9] == "table":
                num_table_annotations += 1

                x0 = min(x0, float(annotations[1]))
                y0 = min(y0, float(annotations[2]))
                x1 = max(x1, float(annotations[3]))
                y1 = max(y1, float(annotations[4]))

    return None if num_table_annotations == 0 else np.array([x0, y0, x1, y1])

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
        annotation_path = doc_root_name + ".txt"
        bounding_box = extract_bounding_box(annotation_path)

        if bounding_box is None:
            continue
        
        image_path = doc_root_name + "_ori.jpg"
        image = extract_image(image_path)

        insertion_index = random.randint(0, len(bounding_boxes))
        images.insert(insertion_index, image)
        bounding_boxes.insert(insertion_index, bounding_box)

    split_index = int(len(bounding_boxes) * train_test_split)

    train_images, train_bounding_boxes = np.array(images[:split_index]), np.array(bounding_boxes[:split_index])
    test_images, test_bounding_boxes = np.array(images[split_index:]), np.array(bounding_boxes[split_index:])
    
    return train_images, train_bounding_boxes, test_images, test_bounding_boxes