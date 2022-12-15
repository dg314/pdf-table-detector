import math


# Resize image dimensions
image_width, image_height = 256, 256


# Anchor box 
anchor_box_scales = [16, 32, 64, 128, 192]
image_ratio = image_height/256
anchor_box_scales = [int(image_ratio*scale) for scale in anchor_box_scales]
# anchor_box_ratios is a list of [width_multiplier, height_multiplier] lists
# anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
anchor_box_ratios = [(1, 1), (1, 2), (math.sqrt(2), 1), (2, 1), (4, 1)]

# RPN Head
rpn_head_activation = 'leaky_relu'


# Weighing the loss functions: [class_loss_wt, regr_loss_wt]
# loss_weights = None
loss_weights = [0.8, 0.2]
loss_weights = [x / sum(loss_weights) for x in loss_weights] # ensuring them sum to 1
