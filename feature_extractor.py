import tensorflow as tf
from stable_baselines.a2c.utils import conv, fc, conv_to_fc

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32)
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def impala_cnn(unscaled_images, depths=[16, 32, 32], use_batch_norm=False, dropout=0.0):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    code from coinrun: https://github.com/openai/coinrun/blob/master/coinrun/policies.py
    """
    #use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        #if Config.DROPOUT > 0:
        if dropout > 0:
            print("DROPOUT HAS NOT BEEN ADDED")
            out_shape = out.get_shape().as_list()

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.get_variable(var_name, shape=batch_seed_shape,
                                         initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None, ...] - dropout))

            curr_mask = curr_mask * (1.0 / (1.0 - dropout))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    #out = images #coinrun scales images outside this function call,
    scaled_images = tf.cast(unscaled_images, tf.float32)
    out = scaled_images
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    #return out, dropout_assign_ops
    return out #not doing dropout
