import tensorflow as tf

# Regular ReLu
def relu(x, name='relu'):
    return tf.maximum(x, 0, name=name);

# Convolutional Layer
# If isFat is true, we can run only with GPUs that have more than 2GB of RAM(with batch size 64)
def conv(input, depth, isTraining=False, isFat=False):
    # filter size 3x3
    output = tf.layers.conv2d(input, depth, [3, 3], strides=(2, 2), padding='SAME')
    output = relu(tf.layers.batch_normalization(output, training=isTraining), name='outputs')

    if isFat:
        # outputs the input dimensions
        output = tf.layers.conv2d(output, depth, [3, 3], strides=(1, 1), padding='SAME')
        output = relu(tf.layers.batch_normalization(output, training=isTraining), name='outputs')

    return output

class Discriminator:

    def __init__(self):
        self.depths = [3, 64, 128, 256, 512]
        self.reuse = False
        self.isFat = False

    # building the Discriminator
    def __call__(self, inputs, training=False, name='D'):
        # convert input to tensor, just for safety
        # 64x64x3 tensor
        input = tf.convert_to_tensor(inputs)

        with tf.name_scope(name), tf.variable_scope('D', reuse=self.reuse):
            # 32x32x64
            with tf.variable_scope('conv_1'):
                output = conv(input, self.depths[1], training, self.isFat)
            # 16x16x128
            with tf.variable_scope('conv_2'):
                output = conv(input, self.depths[2], training, self.isFat)
            # 8x8x256
            with tf.variable_scope('conv_3'):
                output = conv(input, self.depths[3], training, self.isFat)
            # 4x4x512
            with tf.variable_scope('conv_4'):
                outputs = tf.layers.conv2d(output, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # Dense layer for binary classification
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')

        self.reuse = True
        # save variables for regularization
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        return outputs
