# !/usr/bin/env python
# coding: utf-8

import tensorflow as tf


class DualAttentionGuidedDropout(tf.keras.layers.Layer):
    """Dual-attention Guided Dropblock Module for Weakly Supervised Object Localization
    (https://arxiv.org/abs/2003.04719)
    """

    def __init__(self, adl_threshold=0.25, adl_keep_prob=0.90, droprate=0.25, trainable=True, name=None, **kwargs):
        super(DualAttentionGuidedDropout, self).__init__(name=name, trainable=trainable, **kwargs)
        self.drop_prob = 1. - adl_keep_prob
        self.drop_thr = adl_threshold
        self.drop = tf.keras.layers.Dropout(rate=droprate)

    def build(self, input_shape):
        if isinstance(input_shape[3], int):
            self.channels = input_shape[3]
            self.height = input_shape[1]
            self.width = input_shape[2]
        else:
            self.channels = input_shape[3].value
            self.height = input_shape[1].value
            self.width = input_shape[2].value
        super().build(input_shape)

    def get_drop_mask(self, cap_map, drop_thr):
        max_val = tf.reduce_max(cap_map, axis=[1, 2, 3], keepdims=True)
        max_thr_val = max_val * drop_thr
        return tf.cast(cap_map < max_thr_val, dtype=tf.float32) + tf.cast(cap_map > (max_val - max_thr_val),
                                                                          dtype=tf.float32)

    def select_component(self, importance_map, drop_mask, drop_prob):
        random_tensor = tf.random.uniform([], drop_prob, 1. + drop_prob)
        binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def call(self, inputs, training=None):
        if training:
            gap_map = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
            ranking_map, _ = tf.nn.top_k(gap_map, k=self.channels, sorted=True)
            spatial_map = tf.tile((self.drop(ranking_map)), (1, self.height, self.width, 1))
            cap_map = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            importance_map = tf.nn.sigmoid(cap_map)
            drop_mask = self.get_drop_mask(cap_map, self.drop_thr)
            selected_map = self.select_component(importance_map, drop_mask, self.drop_prob)
            output = inputs * selected_map * spatial_map
            return output
        else:
            return inputs
