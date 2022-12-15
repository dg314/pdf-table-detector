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

    def __init__(self, img_w=256, img_h=256, epochs=10, model_weights_save_path='', load_weight_path=None):

        self.resized_image_w = img_w
        self.resized_image_h = img_h
        self.rpn_stride = 16

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = hyperparameters.anchor_box_scales
        # Anchor box ratios
        self.anchor_box_ratios = hyperparameters.anchor_box_ratios
        self.num_anchors = len(self.anchor_box_scales) * len(self.anchor_box_ratios)

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        self.optimizer = 'adam'
        self.rpn_model = None
        self.model_path = None
        self.epochs = epochs
        self.model_weights_save_path = model_weights_save_path

        self.model_initialize()
        # Load weights
        if load_weight_path:
            self.load_model_weights(load_weight_path)

        # Current epoch and time taken
        self.current_epoch = -1
        self.epoch_times = []


    def rpn_heads(self, ft_encoder):
        encoded_conv = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation=hyperparameters.rpn_head_activation, 
                                            kernel_initializer='normal', name='conv_rpn')(ft_encoder)

        rpn_classifier = tf.keras.layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid', 
                                                kernel_initializer='uniform', name='rpn_classify')(encoded_conv)
        rpn_regressor = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', 
                                                kernel_initializer='uniform', name='rpn_regress')(encoded_conv)

        return [rpn_classifier, rpn_regressor, ft_encoder]


    def loss_functions_rpn(self):

        anchor_count = self.num_anchors
        eps = 1e-4
        
        def classification_loss(gt_label, pred_label):
            # gt_label is np.concatenate([is_foreground, has_table], axis=1)
            bce_loss = K.binary_crossentropy(pred_label[:, :, :, :], gt_label[:, :, :, anchor_count:])
            numerator = K.sum(gt_label[:, :, :, :anchor_count] * bce_loss)
            denominator = K.sum(gt_label[:, :, :, :anchor_count] + eps)
            return 1.0 * numerator / denominator

        def regression_loss(gt_label, pred_label):
            # gt_label is np.concatenate([np.repeat(has_table, 4, axis=1), reg_gt], axis=1)
            dists = gt_label[:, :, :, 4 * anchor_count:] - pred_label
            abs_dist = K.abs(dists)
            # If abs_dist <= 1.0, lte1_flag = 1
            lte1_flag = K.cast(K.less_equal(abs_dist, 1.0), tf.float32)

            total_loss = lte1_flag * (0.5 * dists * dists) + (1 - lte1_flag) * (abs_dist - 0.5)
            numerator = K.sum(gt_label[:, :, :, :4 * anchor_count] * total_loss) # consider regr loss only where tables exist
            denominator = K.sum(gt_label[:, :, :, :4 * anchor_count] + eps)
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
                                loss_weights=hyperparameters.loss_weights)
        

    def save_model_weights(self, loss_cls, loss_regr):
        if not os.path.exists(self.model_weights_save_path):
            os.makedirs(self.model_weights_save_path)
        self.rpn_model.save_weights(self.model_weights_save_path)
        with open(self.model_weights_save_path+'loss_cls.pkl', 'wb') as f:
            pickle.dump(loss_cls, f)
        with open(self.model_weights_save_path+'loss_regr.pkl', 'wb') as f:
            pickle.dump(loss_regr, f)


    def load_model_weights(self, load_weight_path):
        try:
            self.rpn_model.load_weights(load_weight_path)
            print("Loaded model weights successfully")
        except Exception as e:
            print(f"Could not load weights: {e}")
