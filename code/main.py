from preprocess import get_data

if __name__ == "__main__":
    train_images, train_bounding_boxes, test_images, test_bounding_boxes = get_data()

    print(f"Training Images Shape: {train_images.shape}")
    print(f"Training Bounding Boxes Shape: {train_bounding_boxes.shape}")
    print(f"Testing Images Shape: {test_images.shape}")
    print(f"Testing Bounding Boxes Shape: {test_bounding_boxes.shape}")