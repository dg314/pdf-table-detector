import numpy as np
from iou import calculate_iou
import random

def calc_rpn(labels,resized_width, resized_height):
    downscale = float(16)
    anchor_sizes = [64, 128, 256]
    anchor_ratios = [[1, 1], [1./np.sqrt(2), 2./np.sqrt(2)], [2./np.sqrt(2), 1./np.sqrt(2)]]  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    w_output = resized_width // 16 #width of output from last layer of VGG16
    h_output = resized_height // 16 #height of output from last layer of VGG16

    n_anchratios = len(anchor_ratios) #num of anchor ratios we are considering 
    num_bboxes = len(labels) #num of bboxes required for either the train or test dataset
  
    has_table = np.zeros((h_output, w_output, num_anchors)) #Boolean-if the there exists a table in the box
    is_foreground = np.zeros((h_output, w_output, num_anchors)) #Boolean- if the box is a foreground box or not
    y_rpn_regr = np.zeros((h_output, w_output, num_anchors * 4)) #stores the trainable targets (inputs for loss function)
  
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)


    gta = np.zeros((num_bboxes, 4))
    for bbox_num, label in enumerate(labels):
        gta[bbox_num, 0] = label[0]
        gta[bbox_num, 1] = label[1]
        gta[bbox_num, 2] = label[2]
        gta[bbox_num, 3] = label[3]

    #Computing anchor boxes using anchor scales and anchor ratios
    for size_idx in range(len(anchor_sizes)):
        for ratio_idx in range(len(anchor_ratios)):
            anchor_x = anchor_sizes[size_idx] * anchor_ratios[ratio_idx][0]
            anchor_y = anchor_sizes[size_idx] * anchor_ratios[ratio_idx][1]
      
            for x in range(w_output):					            
                xmin_anc = downscale * (x + 0.5) - anchor_x / 2
                xmax_anc = downscale * (x + 0.5) + anchor_x / 2	
                    
                # ignore boxes that go across image boundaries
                if xmin_anc < 0 or xmax_anc > resized_width:
                    continue
            
                for y in range(h_output):
					# y-coordinates of the current anchor box
                    ymin_anc = downscale * (y + 0.5) - anchor_y / 2
                    ymax_anc = downscale * (y + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
                    if ymin_anc < 0 or ymax_anc > resized_height:
                        continue

					# bbox_type indicates whether an anchor should be a target
					# Initialize with 'negative'
                    bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0
          
                    for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
                        anch_coords = [xmin_anc, ymin_anc, xmax_anc, ymax_anc]
                        #gt_coords = [gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]]
                        gt_coords = [gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]]
                        curr_iou = calculate_iou(anch_coords,gt_coords)

                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > 0.7:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0 
                            cxa = (xmin_anc + xmax_anc)/2.0
                            cya = (ymin_anc + ymax_anc)/2.0

                            tx = (cx - cxa) / (xmax_anc - xmin_anc)
                            ty = (cy - cya) / (ymax_anc - ymin_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (xmax_anc - xmin_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (ymax_anc - ymin_anc))

                        # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                        if curr_iou > best_iou_for_bbox[bbox_num]:
                            best_anchor_for_bbox[bbox_num] = [jy, ix, ratio_idx, size_idx]
                            best_iou_for_bbox[bbox_num] = curr_iou
                            #best_x_for_bbox[bbox_num,:] = [xmin_anc, xmax_anc, ymin_anc, ymax_anc]
                            best_x_for_bbox[bbox_num,:] = [xmin_anc,  ymin_anc, xmax_anc, ymax_anc]
                            best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                        # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                        if curr_iou > 0.7:
                            bbox_type = 'pos'
                            num_anchors_for_bbox[bbox_num] += 1
                         # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                            if curr_iou > best_iou_for_loc:
                                best_iou_for_loc = curr_iou
                                best_regr = (tx, ty, tw, th)

                        # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                        if 0.3 < curr_iou < 0.7:
                        # gray zone between neg and pos
                            if bbox_type != 'pos':
                                bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        is_foreground[y, x, ratio_idx + n_anchratios * size_idx] = 1
                        has_table[y, x, ratio_idx + n_anchratios * size_idx] = 0
                    elif bbox_type == 'neutral':
                        is_foreground[y, x, ratio_idx + n_anchratios * size_idx] = 0
                        has_table[y, x, ratio_idx + n_anchratios * size_idx] = 0
                    elif bbox_type == 'pos':
                        is_foreground[y, x, ratio_idx + n_anchratios * size_idx] = 1
                        has_table[y, x, ratio_idx + n_anchratios * size_idx] = 1
                        start = 4 * (ratio_idx + n_anchratios * size_idx)
                        y_rpn_regr[y, x, start:start+4] = best_regr

	# we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            is_foreground[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]] = 1
            has_table[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]
  
    has_table = np.transpose(has_table, (2, 0, 1))
    has_table = np.expand_dims(has_table, axis=0)
  
    is_foreground = np.transpose(is_foreground, (2, 0, 1))
    is_foreground = np.expand_dims(is_foreground, axis=0)
    
    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
    
    pos_locs = np.where(np.logical_and(has_table[0, :, :, :] == 1, is_foreground[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(has_table[0, :, :, :] == 0, is_foreground[0, :, :, :] == 1))
    
    num_pos = len(pos_locs[0])
        
    num_regions = 256
    
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        is_foreground[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2
    
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        is_foreground[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
    
    y_rpn_cls = np.concatenate([is_foreground, has_table], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(has_table, 4, axis=1), y_rpn_regr], axis=1)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos