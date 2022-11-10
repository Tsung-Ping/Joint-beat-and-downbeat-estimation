import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import math
import numpy as np
import os
from mir_eval.beat import util

# Disables AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def label2period(y):
    '''y = [B, T], binary matrix with each row indicating a (down)beat sequence.'''
    _, T = tf.unstack(tf.shape(y))
    dist = tf.range(T, dtype=tf.int32) # [T]
    dist = tf.abs(dist[:, None] - dist[None, :]) # [T, T]
    dist = tf.tile(dist[None, :, :], [tf.shape(y)[0], 1, 1]) # [B, T, T]

    mask = tf.cast(tf.equal(y, 0), dtype=tf.int32) * T # [B, T]
    mask = tf.tile(mask[:, None, :], [1, T, 1]) # [B, T, T]
    period = dist + mask
    period = tf.reduce_min(period, axis=2) # [B, T]
    period.set_shape([None, y.get_shape().as_list()[1]])
    return period # [B, T]


def jointly_supervised_class(y1, y2):
    y1_p = tf.not_equal(y1, 0.0) # [B, T]
    y2_p = tf.not_equal(y2, 0.0) # [B, T]
    y_pp = tf.logical_and(y1_p, y2_p) # [B, T], downbeat
    y_np = tf.not_equal(y1_p, y2_p) # [B, T], exclusive beat
    y_nn = tf.logical_not(tf.logical_or(y1_p, y2_p)) # [B, T], non-beat
    y_class = tf.stack([y_pp, y_np, y_nn], axis=2) # [B, T, 3]
    y_class = tf.cast(y_class, tf.float32) # [B, T, 3]
    return y_class


def get_absolute_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.cast(num_timescales, tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal # [1, L, embedding_size]


def normalize(inputs, epsilon=1e-6, adaptive=False, mask=None,
              is_training=None, norm_batch=False, scope="norm", reuse=None):
    '''Applies layer normalization.'''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs.shape.ndims
        reduction_axis = inputs_rank - 1
        moments_axes = list(range(inputs_rank))
        del moments_axes[reduction_axis]
        if not norm_batch:
            del moments_axes[0]
        params_shape = inputs_shape[reduction_axis:reduction_axis + 1]

        if mask is None:
            mean, variance = tf.nn.moments(inputs, axes=moments_axes, keep_dims=True, name='moments')
        else:
            mean, variance = tf.nn.weighted_moments(inputs, axes=moments_axes, frequency_weights=mask, keep_dims=True, name='moments')

        normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
        if not adaptive: # using trainable gain and bias
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer(), trainable=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer(), trainable=True)
            outputs = gamma * normalized + beta
        else: # Adaptive Normalization. ref: Understanding and Improving Layer Normalization (NIPS 2019)
            C = 1
            k = 0.1
            adapter = C*(1 - tf.stop_gradient(k*normalized))
            outputs = adapter * normalized
    return outputs


def generate_relative_positions_embeddings(length_q, length_k, depth, max_relative_position,
                                           learnable=False, name='relative_pos_enc'):
    '''https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py'''
    def _generate_relative_positions_matrix(length_q, length_k, max_relative_position):
        """Generates matrix of relative positions between inputs."""
        if length_q == length_k:
            range_vec_q = range_vec_k = tf.range(length_q)
        else:
            range_vec_k = tf.range(length_k)
            range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat # [l_q, l_k]

    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(length_q, length_k, max_relative_position)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        if learnable:
            embeddings_table = tf.get_variable("rel_pos_embeddings", [vocab_size, depth], trainable=True) # learnable pos embeddings
        else:
            embeddings_table = tf.squeeze(get_absolute_position_encoding(vocab_size, depth), axis=0) # fixed pos embeddings
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings # [l_q, l_k, depth]


def spatial_dropout(x, dropout, is_training, rank, axis):
    noise_shape = [tf.shape(x)[i] if i not in axis else 1 for i in range(rank)]
    return tf.layers.dropout(x, noise_shape=noise_shape, rate=dropout, training=is_training)


def channel_attention(x, reduction=[1,2], gamma=2, b=1, is_training=None, mask=None,
                      highway=None, scope='channel_attention', reuse=None):
    '''ECA-Net - Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2000)'''
    '''x has shape = [B, T, F, C] or [B, T, C]'''
    in_channel = x.get_shape().as_list()[-1]
    k = int(abs((math.log(in_channel, 2) + b) / gamma))
    k = k if k % 2 else k + 1 # make k an odd number
    with tf.variable_scope(scope, reuse=reuse):
        # Global average pooling
        gap = tf.reduce_sum(x*mask, axis=reduction) / tf.reduce_sum(mask, axis=reduction) # global average pooling, [B, C]
        gate = tf.layers.conv1d(inputs=gap[:, :, None], filters=1, kernel_size=k, strides=1, padding='same', name='gap') # [B, C, 1]
        gate = tf.sigmoid(gate)
        gate = tf.squeeze(gate, axis=-1) # [B, C]
        if len(reduction) == 1:
            gate = gate[:, None, :] # [B, 1, C]
        else:
            gate = gate[:, None, None, :] # [B, 1, 1, C]
        out = x * gate
        if highway is not None:
            out += ((1 - gate) * highway)
    return out


def separable_conv1d(x, filters, ksize, dilation=1, stride=1, padding='SAME', is_training=None, dropout=None,
                     symmetric=False, mask=None, scale=None, scope='separable_conv1d', reuse=None):
    '''x has the shape [batch, time, channel]'''
    with tf.variable_scope(scope, reuse=reuse):
        C = x.get_shape().as_list()[-1]

        if symmetric:
            f1 = tf.get_variable(name='d_filter1', shape=(ksize//2+1, 1, C//2, 1), trainable=True) # [kh//2+1, kw, C_in/2, 1]
            f1 = tf.concat([tf.reverse(f1[1:,:,:,:], axis=[0]), f1], axis=0) # [kh, kw, C_in/2, 1]
            f2 = tf.get_variable(name='d_filter2', shape=(ksize, 1, C//2, 1), trainable=True) # [kh, kw, C_in/2, 1]
            d_filter = tf.concat([f1, f2], axis=2) # [kh, kw, C_in, 1]
        else:
            d_filter = tf.get_variable(name='d_filter', shape=(ksize, 1, C, 1), trainable=True) # [kh, kw, C_in, 1]

        # Depthwise
        bias = tf.get_variable(name='bias', shape=(C,), trainable=True)
        conv = tf.nn.depthwise_conv2d(
            input=x[:, :, None, :], filter=d_filter, strides=[1,stride,stride,1], rate=[dilation, 1],
            padding=padding, name='depthwise'
        ) # [B, T, 1, C]
        conv = tf.squeeze(conv, axis=2) + bias # [B, T, C]

        if scale is not None:
            conv += scale

        # Pointwise
        conv = tf.layers.dense(conv, filters, name='pointwise') # [B, T, C]
        conv_mask = tf.ones_like(conv) * mask
        conv = normalize(conv, mask=conv_mask, is_training=is_training, scope='norm_pointwise')
        conv = tf.nn.elu(conv)

        conv = channel_attention(conv, reduction=[1], is_training=is_training, mask=conv_mask, scope='channel_attention') # [B, T, C]
    return conv


def audio_input_encoding_spec(x_spec, filters, activation, dropout, is_training, mask, scope='spec_enc', reuse=None):
    '''x_spec has the shape [batch, n_frames, n_freq=81, channel=2]'''
    with tf.variable_scope(scope, reuse=reuse):
        enc = tf.pad(x_spec, paddings=[(0, 0), (2, 2), (0, 0), (0, 0)], mode='CONSTANT', constant_values=0) # [B, T+4, 81, 2]
        enc1 = tf.layers.conv2d(enc, filters=filters, kernel_size=(5, 41), padding='valid', name='conv11') # [B, T, 41, 24]
        enc2 = tf.layers.conv2d(enc, filters=filters, kernel_size=(5, 61), padding='valid', name='conv12') # [B, T, 21, 24]
        enc = tf.concat([enc1[:, :, 10:-10, :], enc2], axis=-1) # [B, T, 21, 48]
        enc = normalize(enc, mask=mask, is_training=is_training, scope='norm1') # [B, T, 21, 48]
        enc = activation(enc)
        enc = tf.layers.conv2d(
            inputs=enc, filters=filters, kernel_size=(1, 1), padding='same', name='trans1'
        ) # [B, T, 21, 24]

        enc1 = tf.layers.conv2d(enc, filters=filters, kernel_size=(15, 1), padding='same', name='conv21') # [B, T, 21, 24]
        enc2 = tf.layers.conv2d(enc, filters=filters, kernel_size=(21, 1), padding='same', name='conv22') # [B, T, 21, 24]
        enc = tf.concat([enc1, enc2], axis=-1) # [B, T, 21, 48]
        enc = normalize(enc, mask=mask, is_training=is_training, scope='norm2') # [B, T, 21, 48]
        enc = activation(enc)
        enc = spatial_dropout(enc, dropout, is_training, rank=4, axis=[1, 2])

        enc = tf.layers.conv2d(
            inputs=enc, filters=filters, kernel_size=1, padding='same', name='trans2'
        ) # [B, T, 21, 24]

        enc = tf.reduce_max(enc, axis=2) # global max pooling, [B, T, 24]

        # Output
        enc = tf.layers.conv1d(inputs=enc, filters=filters, kernel_size=1, padding='same', name='trans_out') # [B, T, filters]
        enc = normalize(enc, tf.squeeze(mask, axis=-1), is_training=is_training, scope='out_norm') # [B, T, filters]
        enc = activation(enc)
        enc = tf.layers.dropout(enc, rate=dropout, training=is_training) # [B, T, filters]
    return enc


def TCN(x, activation, dropout, is_training, n_layers=4, kernel_size1=301, kernel_size2=401,
        scales = [1,2,3,4], mask=None, scope='TCN', reuse=False):
    '''Temporal Convolutional Network, x shape = [B, T, C], mask = [B, T, 1]'''

    n_scales = len(scales)

    enc = x
    enc_mask = mask[:, :, :, None] # [B, T, 1, 1]
    d_enc = enc.get_shape().as_list()[-1]
    scale_bias1 = tf.get_variable(name='scale_bias1', shape=(n_scales, d_enc), trainable=True)
    scale_bias2 = tf.get_variable(name='scale_bias2', shape=(n_scales, d_enc), trainable=True)
    for i in range(n_layers):
        with tf.variable_scope(scope + '_{}'.format(i), reuse=reuse):
            enc1 = [separable_conv1d(enc,
                                     d_enc,
                                     kernel_size1,
                                     dilation=scales[s],
                                     is_training=is_training,
                                     dropout=dropout,
                                     symmetric=True,
                                     mask=mask,
                                     scale=scale_bias1[s],
                                     scope='enc1',
                                     reuse=tf.AUTO_REUSE) for s in range(n_scales)] # S*[B, T, 24]
            enc2 = [separable_conv1d(enc,
                                     d_enc,
                                     kernel_size2,
                                     dilation=scales[s],
                                     is_training=is_training,
                                     dropout=dropout,
                                     symmetric=True,
                                     mask=mask,
                                     scale=scale_bias2[s],
                                     scope='enc2',
                                     reuse=tf.AUTO_REUSE) for s in range(n_scales)] # S*[B, T, 24]

            # Scale fusion
            enc12 = tf.stack(enc1, axis=3) + tf.stack(enc2, axis=3) # [B, T, 24, S]
            scale_gate = tf.reduce_sum(enc12 * enc_mask, axis=[1, 2], keepdims=True) / tf.reduce_sum(enc_mask, axis=[1, 2], keepdims=True) # GAP, [B, 1, 1, S]
            scale_gate = tf.layers.dense(scale_gate, n_scales, name='scale_gate') # [B, 1, 1, S]
            scale_gate = tf.nn.softmax(scale_gate, axis=3) # [B, 1, 1, S]
            enc12 *= scale_gate # [B, T, 24, S]
            enc12 = tf.reduce_sum(enc12, axis=3) # [B, T, 24]
            enc12 = tf.layers.dense(enc12, d_enc, name='trans') # [B, T, 24]
            enc12 = normalize(enc12, mask=mask, is_training=is_training, scope='norm_part') # [B, T, 24]
            enc12 = activation(enc12)
            enc12 = spatial_dropout(enc12, dropout, is_training, rank=3, axis=[1])
            enc += enc12
    return enc


def audio_input_encoding_baseline(x_spec, filters, activation, dropout, is_training, mask, scope='spec_enc_baseline', reuse=None):
    '''Baleine model: Deconstruct, Analyse, Reconstruct: How to Improve Tempo, Beat, and Downbeat Estimation (ISMIR 2020)'''
    '''x_spec has the shape [batch, n_frames, n_freq=81, channel=2]'''

    with tf.variable_scope(scope, reuse=reuse):
        # Convolution
        enc = tf.pad(x_spec, paddings=[(0, 0), (2, 2), (0, 0), (0, 0)], mode='CONSTANT') # [B, T+4, 81, 2]

        enc = tf.layers.conv2d(
            inputs=enc, filters=20, kernel_size=(3, 3), padding='valid', name='conv1', activation=activation
        ) # [B, T+2, 79, 20]
        enc = tf.layers.max_pooling2d(
            inputs=enc, pool_size=(1, 3), strides=(1, 3), padding='valid', name='mp1'
        ) # [B, T+2, 26, 20]
        enc = tf.layers.dropout(enc, rate=dropout, training=is_training, name='drop1') # [B, T+2, 79, 20]

        enc = tf.layers.conv2d(
            inputs=enc, filters=20, kernel_size=(1, 12), padding='valid', name='conv2', activation=activation
        ) # [B, T+2, 15, 20]
        enc = tf.layers.max_pooling2d(
            inputs=enc, pool_size=(1, 3), strides=(1, 3), padding='valid', name='mp2'
        ) # [B, T+2, 5, 20]
        enc = tf.layers.dropout(enc, rate=dropout, training=is_training, name='drop2') # [B, T+2, 15, 20]

        enc = tf.layers.conv2d(
            inputs=enc, filters=20, kernel_size=(3, 3), padding='valid', name='conv3', activation=activation
        ) # [B, T, 3, 20]
        enc = tf.layers.max_pooling2d(
            inputs=enc, pool_size=(1, 3), strides=(1, 3), padding='valid', name='mp3'
        ) # [B, T, 1, 20]
        enc = tf.layers.dropout(enc, rate=dropout, training=is_training, name='drop3') # [B, T, 3, 20]

        # Output
        enc = tf.squeeze(enc, axis=2) # [B, T, 20]
        enc = tf.layers.dense(enc, filters, name='concat_dense') # [B, T, filters]
        enc = normalize(enc, mask, is_training=is_training, scope='norm_enc') # [B, T, filters]
    return enc


def TCN_baseline(x, activation, dropout, is_training, n_layers=11, mask=None, scope='TCN_baseline', reuse=False):
    '''Baseline model: Temporal Convolutional Networks for Musical Audio Beat Tracking (EUSIPCO 2019)'''
    '''Temporal Convolutional Network, x shape = [B, T, C], mask = [B, T, 1]'''
    enc = x
    C = x.get_shape().as_list()[-1]
    for i in range(n_layers):
        with tf.variable_scope(scope + '_{}'.format(i), reuse=reuse):
            # Dilated convolution
            enc1 = tf.layers.conv1d(
                inputs=enc, filters=C, kernel_size=5, padding='same', dilation_rate=2**i,
                activation=activation, name='di_conv1_' + '{}'.format(i)
            ) # [B, T, f]

            enc2 = tf.layers.conv1d(
                inputs=enc, filters=C, kernel_size=5, padding='same', dilation_rate=2**(i+1),
                activation=activation, name='di_conv2_' + '{}'.format(i)
            ) # [B, T, f]

            enc = tf.concat([enc1, enc2], axis=-1) # [B, T, 2f]
            enc = spatial_dropout(enc, dropout, is_training, rank=3, axis=[1]) # [B, T, 2f]

            enc = tf.layers.dense(enc, C, name='out_dense') # [B, T, f]
            enc = normalize(enc, mask=mask, is_training=is_training, scope='norm_enc') # [B, T, C]
    return enc


def audio_joint_beat_downbeat_estimation_baseline(x_spec, x_len, dropout, is_training, hp):
    '''x_spec has shape = [batch_size, time, n_mels, 2]'''
    with tf.variable_scope("input_encoding"):
        seq_mask = tf.sequence_mask(x_len, maxlen=hp.sequence_length, dtype=tf.float32) # [B, T]
        enc = audio_input_encoding_baseline(
            x_spec, hp.n_filters, hp.activation, dropout, is_training, seq_mask[:, :, None,], scope='enc_baseline'
        ) # [B, T, C]

    with tf.variable_scope("Sequunece_modeling"):
        h = TCN_baseline(
            enc, n_layers=11, activation=hp.activation,
            dropout=dropout, is_training=is_training, mask=seq_mask[:, :, None], scope='TCN_baseline') # [B, T, C]

    with tf.variable_scope('output'):
        logits_db = tf.squeeze(tf.layers.dense(h, 1, name='dense_db'), axis=-1) # [B, T]
        logits_b = tf.squeeze(tf.layers.dense(h, 1, name='dense_b'), axis=-1) # [B, T]
    return logits_db, logits_b


def label_expanding(y, expanding_sizes, epsilon=0.5, mask=None):
    '''y = [B, T], expanding_sizes = [B]'''
    y = tf.cast(y, tf.float32)
    max_size = tf.reduce_max(expanding_sizes)
    right = tf.sequence_mask(expanding_sizes, maxlen=max_size, dtype=tf.bool) # [B, max_size]
    left = tf.reverse(right, axis=[1]) # [B, max_size]
    center = tf.ones_like(expanding_sizes, dtype=tf.bool)[:, None] # [B, 1]
    kernel = tf.concat([left, center, right], axis=-1) # [B, 2*max_size+1]
    filter = tf.transpose(kernel[:, :, None, None], [1,2,0,3]) # [2*max_size+1, 1, B, 1]
    filter = tf.cast(filter, tf.float32)
    y_ = tf.transpose(y[:, :, None, None], [2,1,3,0]) # [1, T, 1, B]
    y_expand = tf.nn.depthwise_conv2d(y_, filter=filter, strides=[1,1,1,1], padding='SAME', data_format='NHWC') # [1, T, 1, B]
    y_expand = tf.squeeze(y_expand, axis=[0,2]) # [T, B]
    y_expand = tf.transpose(y_expand, [1,0]) # [T, B]
    y_expand = (1-epsilon)*y + epsilon*y_expand
    if mask is not None:
        y_expand *= mask
    return y_expand


def output_probability_pooling(prob, pool_size=13, strides=1):
    '''prob is a 2D tensor containing probabilities.'''
    prob_pool = tf.layers.max_pooling1d(prob[:,:,None], pool_size=pool_size, strides=strides, padding='same')
    prob_pool = tf.squeeze(prob_pool, axis=2)
    return tf.where(tf.equal(prob_pool, prob), prob, tf.zeros_like(prob))


def audio_joint_beat_downbeat_estimation_classTCN(x_spec, x_len, dropout, is_training, hp):
    '''x_spec has shape = [batch_size, time, n_mels, 6]'''
    with tf.variable_scope("input_encoding"):
        seq_mask = tf.sequence_mask(x_len, maxlen=hp.sequence_length, dtype=tf.float32) # [B, T]
        enc = audio_input_encoding_spec(
            x_spec, hp.n_filters, hp.activation, dropout, is_training, seq_mask[:, :, None, None], scope='enc'
        ) # [B, T, C]

    with tf.variable_scope("Sequunece_modeling"):
        h = TCN(
            enc, activation=hp.activation, dropout=dropout, is_training=is_training, mask=seq_mask[:, :, None], scope='TCN'
        ) # [B, T, C]
    return h


def symbolic_encoder(x_pianoroll, x_onset, x_IOI, x_flux, filters, activation, mask,
                     dropout, is_training, scope='musicnet_encoder', reuse=None):
    '''
    x_pianoroll = [batch, n_frames, 88]
    x_onset = [batch, n_frames, 88]
    x_IOI = [batch, n_frames]
    x_flux = [batch, n_frames]
    '''

    x_po = tf.stack([x_pianoroll, x_onset], axis=3, name='po_stack') # [B, T, 88, 2]
    x_if = tf.stack([x_IOI, x_flux], axis=2, name='if_stack') # [B, T, 2]

    # Input norm
    x_po = tf.layers.batch_normalization(x_po, training=is_training, name="bn_po") # [B, T, 88, 2]
    x_if = tf.layers.batch_normalization(x_if, training=is_training, name="bn_if") # [B, T, 2]

    with tf.variable_scope(scope, reuse=reuse):
        # Pianoroll & Onset
        enc_po = tf.layers.conv2d(inputs=x_po, filters=4, kernel_size=(51, 1), padding='same', name='conv_po1') # [B, T, 88, 4]
        enc_po = tf.layers.conv2d(inputs=enc_po, filters=8, kernel_size=(1, 23), padding='valid', name='conv_po2') # [B, T, 66, 8]
        enc_po = activation(enc_po) # [B, T, 66, 8]
        enc_po = tf.transpose(enc_po, [0, 3, 1, 2]) # [B, 8, T, 66]
        _, C, T, F = enc_po.get_shape().as_list()

        enc_po = tf.pad(x_po, paddings=[(0, 0), (2, 2), (0, 0), (0, 0)], mode='CONSTANT', constant_values=0) # [B, T+4, 88, 2]
        enc_po1 = tf.layers.conv2d(enc_po, filters=filters, kernel_size=(5, 41), padding='valid', name='conv11') # [B, T, 48, 20]
        enc_po2 = tf.layers.conv2d(enc_po, filters=filters, kernel_size=(5, 61), padding='valid', name='conv12') # [B, T, 28, 20]
        enc_po = tf.concat([enc_po1[:, :, 10:-10, :], enc_po2], axis=-1) # [B, T, 28, 40]
        enc_po = normalize(enc_po, mask=mask, is_training=is_training, scope='norm1') # [B, T, 28, 40]
        enc_po = activation(enc_po)
        enc_po = tf.layers.conv2d(
            inputs=enc_po, filters=filters, kernel_size=(1, 1), padding='same', name='trans1'
        ) # [B, T, 28, 20]

        enc_po1 = tf.layers.conv2d(enc_po, filters=filters, kernel_size=(15, 1), padding='same', name='conv21') # [B, T, 28, 20]
        enc_po2 = tf.layers.conv2d(enc_po, filters=filters, kernel_size=(21, 1), padding='same', name='conv22') # [B, T, 28, 20]
        enc_po = tf.concat([enc_po1, enc_po2], axis=-1) # [B, T, 28, 40]
        enc_po = normalize(enc_po, mask=mask, is_training=is_training, scope='norm2') # [B, T, 28, 40]
        enc_po = activation(enc_po)
        enc_po = spatial_dropout(enc_po, dropout, is_training, rank=4, axis=[1, 2])

        enc_po = tf.layers.conv2d(
            inputs=enc_po, filters=filters, kernel_size=1, padding='same', name='trans2'
        ) # [B, T, 28, 20]

        enc_po = tf.reduce_max(enc_po, axis=2) # global max pooling, [B, T, 20]

        # IOI & Flux
        enc_if = tf.layers.conv1d(inputs=x_if, filters=4, kernel_size=51, padding='same', name='conv_if') # [B, T, 4]
        enc_if = activation(enc_if)

        # Output
        enc = tf.concat([enc_po, enc_if], axis=-1, name='enc_concat')  # [B, T, 24]
        enc = tf.layers.conv1d(inputs=enc, filters=filters, kernel_size=1, padding='same', name='trans_out') # [B, T, filters]
        enc = normalize(enc, tf.squeeze(mask, axis=-1), is_training=is_training, scope='out_norm') # [B, T, filters]
        enc = activation(enc)
        enc = tf.layers.dropout(enc, rate=dropout, training=is_training) # [B, T, filters]
    return enc


def symbolic_joint_beat_downbeat_estimation_classTCN(x_pianoroll, x_onset, x_IOI, x_flux, x_len, dropout, is_training, hp):
    '''x_pianoroll with shape = [batch, seq_len, 88]
    x_onset with shape = [batch, seq_len, 88]
    x_IOI with shape = [batch, seq_len]'''
    with tf.variable_scope("input_encoding"):
        seq_mask = tf.sequence_mask(x_len, maxlen=hp.sequence_length, dtype=tf.float32) # [B, T]
        enc = symbolic_encoder(x_pianoroll, x_onset, x_IOI, x_flux, hp.n_filters, hp.activation,
                                   seq_mask[:, :, None, None], dropout, is_training, scope='encoder') # [B, T, f]

    with tf.variable_scope("Sequunece_modeling"):
        h = TCN(
            enc, activation=hp.activation, dropout=dropout, is_training=is_training, mask=seq_mask[:, :, None], scope='TCN'
        ) # [B, T, C]
    return h


# auido sampling rate
audio_sr = 44100


def sequence_recovery(sequences, rids, lens, resolution=None):
    op_dict = {}
    for seq, rid, len in zip(sequences, rids, lens):
        key = (rid).split(':')[0]
        if resolution is None:
            global audio_sr
            resolution = int(key.split('_hop=')[1]) / audio_sr

        if key not in op_dict.keys():
            op_dict[key] = []

        op_dict[key].append(seq[:len])

    for key, value in op_dict.items():
        recovered_seqeucne = np.concatenate(value)
        time = np.array([i for i, x in enumerate(recovered_seqeucne) if x == 1]) * resolution # in sec
        op_dict[key] = time
    return op_dict


def f_measure(reference_beats, estimated_beats, f_measure_threshold):
    """Compute the F-measure of correct vs incorrectly predicted beats.
    "Correctness" is determined over a small window.
    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    f_measure_threshold : float
        Window size, in seconds
        (Default value = 0.07)
    Returns
    -------
    A tuple of:
        precision: float
        recall: float
        f_score : float
        n_beats: int
    """
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return (0, 0, 0, len(estimated_beats), len(reference_beats)) # (P, R, F1, len(pred), len(label))
    # Compute the best-case matching between reference and estimated locations
    matching = util.match_events(reference_beats,
                                 estimated_beats,
                                 f_measure_threshold)

    precision = len(matching) / len(estimated_beats)
    recall = len(matching) / len(reference_beats)
    F1 = util.f_measure(precision, recall)
    precision = np.round(precision, 4)
    recall = np.round(recall, 4)
    F1 = np.round(F1, 4)
    return (precision, recall, F1, len(estimated_beats), len(reference_beats))


def beat_eval(pred, label_dict, f_measure_threshold=0.07, resolution=None, target=None):

    if target == 'downbeat':
        label_op_dict = sequence_recovery(label_dict['downbeat'], label_dict['recovery_id'], label_dict['len'], resolution=resolution)
    elif target == 'beat':
        label_op_dict = sequence_recovery(label_dict['beat'], label_dict['recovery_id'], label_dict['len'], resolution=resolution)
    else:
        print('Error: invalid target.')
        exit()
    pred_op_dict = sequence_recovery(pred, label_dict['recovery_id'], label_dict['len'], resolution=resolution)

    score_dict = {key: f_measure(reference_beats=label_op_dict[key],
                                 estimated_beats=pred_op_dict[key],
                                 f_measure_threshold=f_measure_threshold) for key in label_op_dict.keys()}
    return score_dict, label_op_dict, pred_op_dict # {key: (P, R, F1, n_beats)...}


def focal_loss_from_probs(labels, p, gamma=2, alpha=0.75, epsilon=1e-7, mask=None, weight=None, reduce=True):
    """Compute focal loss from probabilities. ref.: https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_binary_focal_loss.py
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    p : tf.Tensor
        Estimated probabilities for the positive class.
    gamma : float
        Focusing parameter.
    alpha : float, weighting factor, [0,1]
        If not None, losses for the positive class will be scaled by this
        weight.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """
    # Predicted probabilities for the negative class

    q = (1 - p) # [B, T]
    labels_pos = tf.cast(labels, tf.float32)
    labels_neg = (1 - labels_pos)

    # Loss for the positive examples
    loss_pos = -alpha * (q**gamma) * tf.log(p + epsilon) # [B, T]

    # Loss for the negative examples
    loss_neg = -(1-alpha) * (p**gamma) * tf.log(q + epsilon) # [B, T]

    # Combine loss terms
    loss = labels_pos * loss_pos + labels_neg * loss_neg # [B, T]

    if weight is not None:
        weight = tf.stop_gradient(weight)
        loss *= weight # [B, T]

    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask) if reduce else loss


def dice_loss_from_probs(labels, p, mask=None, epsilon=1e-5, axis=[1]):
    '''labels, p, and mask with shape = [B, T] or [B, T, C]'''
    if mask is not None:
        p *= mask
        labels *= mask

    numerator = 2 * tf.reduce_sum(p*labels, axis=axis, name='sum_pq') + epsilon # [B]
    denominator = tf.reduce_sum(p**2, axis=axis, name='sum_pp') + tf.reduce_sum(labels**2, axis=axis, name='sum_qq') + epsilon # [B]
    dice_coef = numerator / denominator # [B]
    pos_loss = -tf.log(dice_coef) # [B]
    loss = tf.reduce_mean(pos_loss) #+ neg_loss
    return loss


def categorical_focal_loss_from_probs(labels, probs, gamma=2, alpha=[4, 3, 1], epsilon=1e-7, label_smoothing=0.0, mask=None, reduce=True):
    """Compute focal loss from probabilities. ref.: https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_binary_focal_loss.py"""
    # labels = [B, T, Classes], one-hot
    # probs = [B, T, Classes]
    # mask = [B, T]

    if label_smoothing > 0.0:
        labels = labels*(1 - label_smoothing) + (1 - labels)*(label_smoothing / (labels.get_shape().as_list()[-1] - 1))

    p = probs # [B, T, C]
    q = 1 - probs # [B, T, C]
    alpha = tf.constant(alpha, dtype=tf.float32)[None, None, :] # [1, 1, C]
    alpha /= tf.reduce_sum(alpha, keepdims=True) # normalization, [1, 1, C]
    ce = -tf.log(p + epsilon) # [B, T, C]
    w = alpha * (q**gamma) # [B, T, C]
    loss = labels * w * ce # [B, T, C]
    loss = tf.reduce_sum(loss, axis=-1) # [B, T]
    return tf.reduce_sum(loss*mask) / tf.reduce_sum(mask) if reduce else loss