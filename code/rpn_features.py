import numpy as np

def calc_rpn(image_data, labels, width, height, resized_width, resized_height):
    downscale = float(16)
    anchor_sizes = [64, 128, 256]
    anchor_ratios = [[1, 1], [1./np.sqrt(2), 2./np.sqrt(2)], [2./np.sqrt(2), 1./np.sqrt(2)]]  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    w_output = resized_width // 16 #(??????????????????) #width of output 
    h_output = resized_height // 16 #(??????????????????) #height of output 

    n_anchratios = len(anchor_ratios) #number of different anchor ratios we are considering 
    num_bboxes = len(labels) #number of bboxes required for either the train or test dataset
  