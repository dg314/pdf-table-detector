from preprocess import get_data
from visualize import visualize_doc

visualize_all = False

if __name__ == "__main__":
    train_images, train_bounding_boxes, test_images, test_bounding_boxes = get_data()

    print(f"Training Images Shape: {train_images.shape}")
    print(f"Training Bounding Boxes Shape: {train_bounding_boxes.shape}")
    print(f"Testing Images Shape: {test_images.shape}")
    print(f"Testing Bounding Boxes Shape: {test_bounding_boxes.shape}")

    if visualize_all:
        for train_image, train_bounding_box in zip(train_images, train_bounding_boxes):
            visualize_doc(train_image, train_bounding_box)

        for test_image, test_bounding_box in zip(test_images, test_bounding_boxes):
            visualize_doc(test_image, test_bounding_box)