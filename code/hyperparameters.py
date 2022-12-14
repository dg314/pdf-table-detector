import math


# Resize image dimensions
image_width, image_height = 256, 256


# Anchor box 
anchor_box_scales = [16, 32, 64, 128, 192]
# [64, 128, 256] are appropriate for a 256x256 image
image_ratio = image_height/256
anchor_box_scales = [int(image_ratio*scale) for scale in anchor_box_scales]
# anchor_box_ratios is a list of [width_multiplier, height_multiplier] lists
# anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
anchor_box_ratios = [[1, 1], [2, 1], [1, math.sqrt(2)], [1, 2], [1, 4]]


# RPN Head
rpn_head_activation = 'leaky_relu'

