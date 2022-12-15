import pandas as pd
import numpy as np
from iou import compute_iou


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]


    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    return boxes


def match_regions(y_true, y_predicted, threshold=0.3):
    hits, accuracy = [], []
    for true_roi in y_true:
        for pred_roi in y_predicted:
            match = compute_iou(true_roi, pred_roi)
            if match > threshold:
                hits.append(pred_roi)
                accuracy.append(match)
    
    hits = [x for x, _ in sorted(zip(hits, accuracy), key=lambda pair: pair[1], reverse=True)]
    accuracy = sorted(accuracy, reverse=True)
    if len(accuracy) > 0:
        mean_acc = np.mean(accuracy)
    else:
        mean_acc = 0
    
    return hits, mean_acc


def regress_anchor(unadjusted_box, adjustment):
    try:
        dx = adjustment[0, :, :]
        dy = adjustment[1, :, :]
        dw = adjustment[2, :, :]
        dh = adjustment[3, :, :]
        x0 = unadjusted_box[0, :, :]
        y0 = unadjusted_box[1, :, :]
        w0 = unadjusted_box[2, :, :]
        h0 = unadjusted_box[3, :, :]

        # (cx, cy) = center of box
        cx0 = x0 + w0/2.
        cy0 = y0 + h0/2.
        # (cx1, cy1) = regressed (adjusted) center of box
        adj_cx = dx * w0 + cx0
        adj_cy = dy * h0 + cy0

        # adj_w, adj_h = regressed width and height of box
        adj_w = np.exp(dw.astype(np.float64)) * w0
        adj_h = np.exp(dh.astype(np.float64)) * h0
        # (adj_x, adj_y) = regressed top left of box
        adj_x = adj_cx - adj_w/2.
        adj_y = adj_cy - adj_h/2.

        return np.round(np.stack([adj_x, adj_y, adj_w, adj_h]))
    except:
        # Return the un-adjusted coordinates if error
        return unadjusted_box


# RPN to ROI
def infer_roi_from_rpn(anchor_sizes, anchor_ratios, classif_layer, regress_layer, max_boxes=10, overlap_thresh=0.9):
    feature_map_height = classif_layer.shape[1]
    feature_map_width = classif_layer.shape[2]
    anchor_count = classif_layer.shape[3]

    # TODO:
    regress_layer = regress_layer / 4.0 # WHY IS THIS DONE???
    layer_ix = 0
    roi_proposals = np.zeros((4, feature_map_height, feature_map_width, anchor_count)) 

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            anchor_width = (anchor_ratio[0] * anchor_size) / 16
            anchor_height = (anchor_ratio[1] * anchor_size) / 16
            
            # Regress the current (size, shape) box for each position
            curr_box_regression = regress_layer[0, :, :, (layer_ix*4):((layer_ix*4) + 4)].transpose(2, 0, 1)
            
            # Create a grid for the anchor box centers
            grid_X, grid_Y = np.meshgrid(np.arange(feature_map_width), np.arange(feature_map_height))

            # Defining ROI proposals by top left, width and height
            roi_proposals[0, :, :, layer_ix] = grid_X - anchor_width / 2
            roi_proposals[1, :, :, layer_ix] = grid_Y - anchor_height / 2
            roi_proposals[2, :, :, layer_ix] = anchor_width
            roi_proposals[3, :, :, layer_ix] = anchor_height

            roi_proposals[:, :, :, layer_ix] = regress_anchor(roi_proposals[:, :, :, layer_ix], curr_box_regression)

            # Discard if width or height > 1
            roi_proposals[2, :, :, layer_ix] = np.maximum(1, roi_proposals[2, :, :, layer_ix])
            roi_proposals[3, :, :, layer_ix] = np.maximum(1, roi_proposals[3, :, :, layer_ix])

            # Transform coordinates to the (xmin, ymin, xmax, ymax) system
            roi_proposals[2, :, :, layer_ix] += roi_proposals[0, :, :, layer_ix]
            roi_proposals[3, :, :, layer_ix] += roi_proposals[1, :, :, layer_ix]

            # Discard boxes beyond the boundaries of feature map
            roi_proposals[0, :, :, layer_ix] = np.maximum(0, roi_proposals[0, :, :, layer_ix])
            roi_proposals[1, :, :, layer_ix] = np.maximum(0, roi_proposals[1, :, :, layer_ix])
            roi_proposals[2, :, :, layer_ix] = np.minimum(feature_map_width-1, roi_proposals[2, :, :, layer_ix])
            roi_proposals[3, :, :, layer_ix] = np.minimum(feature_map_height-1, roi_proposals[3, :, :, layer_ix])

            layer_ix += 1

    roi_proposed_boxes = roi_proposals.transpose((0, 3, 1, 2)).reshape((4, -1)).transpose()
    roi_proposed_box_probs = classif_layer.transpose((0, 3, 1, 2)).flatten()

    roi_valid_proposed_boxes = roi_proposed_boxes[np.where((roi_proposed_boxes[:, 0] < roi_proposed_boxes[:, 2]) & \
                                                            (roi_proposed_boxes[:, 1] < roi_proposed_boxes[:, 3]))]
    roi_valid_proposed_box_probs = roi_proposed_box_probs[np.where((roi_proposed_boxes[:, 0] < roi_proposed_boxes[:, 2]) & \
                                                                    (roi_proposed_boxes[:, 1] < roi_proposed_boxes[:, 3]))]

    # Discard non-maxes
    chosen_bboxes = non_max_suppression_fast(roi_valid_proposed_boxes, roi_valid_proposed_box_probs, 
                                            overlap_thresh=overlap_thresh, max_boxes=max_boxes)

    return chosen_bboxes

