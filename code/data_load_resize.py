
import cv2
import numpy as np

def data_loading_and_resizing(image_list, train_bounding_boxes):
    images = []
    labels = []

    for i in range(len(image_list)):
        x_scale = 256 / image_list[i].shape[0]
        y_scale = 256 / image_list[i].shape[1]
        
        # Resize Images to 256px x 256px
        image = cv2.resize(image_list[i], (256,256)) 
        
        # Resize the bbox coordinates
        if not np.any(train_bounding_boxes[i]<0):
            xmin = int(np.round(train_bounding_boxes[i][0] * x_scale))
            ymin = int(np.round(train_bounding_boxes[i][1] * y_scale))
            xmax = int(np.round(train_bounding_boxes[i][2] * x_scale))
            ymax = int(np.round(train_bounding_boxes[i][3] * y_scale))

            bboxes = [xmin,ymin,xmax,ymax]
        else:
            # This will be [-1, -1, -1, -1]
            bboxes = train_bounding_boxes[i].tolist()

        labels.append(bboxes)
        images.append(image)

    return images, labels