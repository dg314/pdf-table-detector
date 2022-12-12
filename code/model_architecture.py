import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
import tensorflow.keras.backend as K
import numpy as np
import math
# from keras.backend import image_dim_ordering
from keras.backend import image_data_format

class ModelConfig:

    def __init__(self):

        self.rpn_stride = 16

        self.resized_image_w = 256
        self.resized_image_h = 256

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = [64, 128, 256] 

        # Anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

        self.num_anchors = len(self.anchor_box_scales) * len(self.anchor_box_ratios)

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 300

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        self.optimizer = 'adam'
        self.rpn_model = None
        self.model_path = None


    def rpn_heads(self, ft_encoder):
        encoded_conv = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', 
                                        kernel_initializer='normal', name='conv_rpn')(ft_encoder)

        rpn_classifier = tf.keras.layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid', 
                                                kernel_initializer='uniform', name='rpn_classify')(encoded_conv)
        rpn_regressor = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', 
                                                kernel_initializer='uniform', name='rpn_regress')(encoded_conv)

        return [rpn_classifier, rpn_regressor, ft_encoder]


    def loss_functions_rpn(self):

        anchor_count = self.num_anchors
        eps = 1e-4
        
        def classification_loss(y_true, y_pred):

            numerator = K.sum(y_true[:, :, :, :anchor_count] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, anchor_count:]))
            denominator = K.sum(eps + y_true[:, :, :, :anchor_count])

            return 1.0 * numerator / denominator

        def regression_loss(y_true, y_pred):

            # x is the difference between true value and predicted vaue
            x = y_true[:, :, :, 4 * anchor_count:] - y_pred
            x_abs = K.abs(x)

            # If x_abs <= 1.0, x_bool = 1
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            numerator = K.sum(y_true[:, :, :, :4 * anchor_count] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
            denominator = K.sum(eps + y_true[:, :, :, :4 * anchor_count])

            return 1.0 * numerator / denominator

        return [classification_loss, regression_loss]


    def model_initialize(self):
        vgg16 = tf.keras.applications.vgg16.VGG16(input_shape=(self.resized_image_w, self.resized_image_h, 3),
                                                                weights='imagenet', include_top=False)
        resized_image = tf.keras.layers.Input(shape=(None,None,3))
        feature_encoder  = tf.keras.models.Model(inputs=vgg16.get_layer(index=0).input, 
                                                        outputs=vgg16.get_layer(index=-2).output)
        feature_encoder_layer = feature_encoder(resized_image)
        rpn_heads = self.rpn_heads(feature_encoder_layer)
        loss_funcs = self.loss_functions_rpn()

        self.rpn_model = tf.keras.models.Model(resized_image, rpn_heads[:2])
        self.rpn_model.compile(optimizer=self.optimizer, 
                                loss=[loss_funcs[0], loss_funcs[1]],
                                metrics=["acc"])
        
