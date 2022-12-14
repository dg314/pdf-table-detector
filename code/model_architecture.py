import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
import tensorflow.keras.backend as K
import numpy as np
import math
# from keras.backend import image_dim_ordering
from keras.backend import image_data_format
import os
import time
import pickle
import hyperparameters


class ModelConfig:

    def __init__(self, img_w=256, img_h=256, epochs=20, model_weights_save_path='', load_weight_path=None):

        self.rpn_stride = 16

        self.resized_image_w = img_w
        self.resized_image_h = img_h

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = hyperparameters.anchor_box_scales

        # Anchor box ratios
        self.anchor_box_ratios = hyperparameters.anchor_box_ratios

        self.num_anchors = len(self.anchor_box_scales) * len(self.anchor_box_ratios)

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 300

        # overlaps for RPN
        self.rpn_min_overlap = 0.5
        self.rpn_max_overlap = 0.9

        self.optimizer = 'adam'
        self.rpn_model = None
        self.model_path = None
        self.epochs = epochs

        # Initialize the model
        self.model_initialize()

        self.model_weights_save_path = model_weights_save_path

        # Load weights
        if load_weight_path:
            self.load_model_weights(load_weight_path)

        # Current epoch and time taken
        self.current_epoch = -1
        self.epoch_times = []


    def rpn_heads(self, ft_encoder):
        encoded_conv = tf.keras.layers.Conv2D(512, 3, (1, 1), padding='same', activation=hyperparameters.rpn_head_activation, 
                                            kernel_initializer='normal', name='conv_rpn')(ft_encoder)

        # Org
        rpn_classifier = tf.keras.layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid', 
                                                kernel_initializer='uniform', name='rpn_classify')(encoded_conv)
        rpn_regressor = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', 
                                                kernel_initializer='uniform', name='rpn_regress')(encoded_conv)

        # More complex
        # dense1 = tf.keras.layers.Dense(256, activation=hyperparameters.rpn_head_activation)(encoded_conv)
        # rpn_classifier = tf.keras.layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid', 
        #                                         kernel_initializer='uniform', name='rpn_classify')(dense1)
        # rpn_regressor = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', 
        #                                         kernel_initializer='uniform', name='rpn_regress')(dense1)

        # Most complex
        # dense1 = tf.keras.layers.Dense(256, activation=hyperparameters.rpn_head_activation)(encoded_conv)
        # dense2 = tf.keras.layers.Dense(100, activation=hyperparameters.rpn_head_activation)(dense1)
        # dense3 = tf.keras.layers.Dense(55, activation=hyperparameters.rpn_head_activation)(dense2)
        # dense4 = tf.keras.layers.Dense(15, activation=hyperparameters.rpn_head_activation)(dense3)
        # rpn_classifier = tf.keras.layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid', 
        #                                         kernel_initializer='uniform', name='rpn_classify')(dense4)
        # rpn_regressor = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', 
        #                                         kernel_initializer='uniform', name='rpn_regress')(dense4)
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
        

    def save_model_weights(self, loss_cls, loss_regr, accuracy):
        if not os.path.exists(self.model_weights_save_path):
            os.makedirs(self.model_weights_save_path)
        self.rpn_model.save_weights(self.model_weights_save_path)
        with open(self.model_weights_save_path+'loss_cls.pkl', 'wb') as f:
            pickle.dump(loss_cls, f)
        with open(self.model_weights_save_path+'loss_regr.pkl', 'wb') as f:
            pickle.dump(loss_regr, f)
        with open(self.model_weights_save_path+'accuracy.pkl', 'wb') as f:
            pickle.dump(accuracy, f)


    def load_model_weights(self, load_weight_path):
        try:
            self.rpn_model.load_weights(self.model_weights_save_path)
            print("Loaded model weights successfully")
        except Exception as e:
            print(f"Could not load weights: {e}")
