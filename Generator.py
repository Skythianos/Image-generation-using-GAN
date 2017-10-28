import tensorflow as tf

# LeakyReLu
def leaky_relu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak*x, name=name)

# Deconvolutional layer
# If isFat is true, we can run only with GPUs that have more than 2GB of RAM(with batch size 64)
def deconv(input, depth, isTraining=False, isFat=False):
    # filter size is 3x3
    output = tf.layers.conv2d_transpose(input, depth, [3, 3], strides=(2, 2), padding='SAME')
    output = leaky_relu(tf.layers.batch_normalization(output, training=isTraining), name='outputs')

    if isFat:
        # outputs the input dimensions
        output = tf.layers.conv2d_transpose(output, depth, [3, 3], strides=(1, 1), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=isTraining), name='outputs')

    return output


class Generator:
    def __init__(self, s_size=4, isFat=False):
        # depths represent the length of the filterbacth
        self.depths =  depths=[1024, 512, 256, 128, 3]
        self.s_size = s_size #initial dimension size
        self.reuse = False
        self.isFat = isFat

    #building the model
    def __call__(self, inputs, training=False):
        # converting input data to tensor
        input = tf.convert_to_tensor(inputs)
        with tf.variable_scope('G', reuse=self.reuse):
            # reshape input with dense layer, this also provide trainable parameters 4x4x1024
            with tf.variable_scope('reshape'):
                output = tf.layers.dense(input, self.depths[0] * self.s_size * self.s_size)
                output = tf.reshape(output, [-1, self.s_size, self.s_size, self.depths[0]])
                output = leaky_relu(tf.layers.batch_normalization(output, training=training), name='outputs')
            # first deconv layer 8x8x512
            with tf.variable_scope('deconv_1'):
                output = deconv(output,self.depths[1],training,self.isFat)
            # second deconv layer 16x16x256
            with tf.variable_scope('deconv_2'):
                output = deconv(output,self.depths[2],training,self.isFat)
            # third deconv layer 32x32x128
            with tf.variable_scope('deconv_3'):
                output = deconv(output, self.depths[3], training, self.isFat)
            # fourt deconv layer 64x64x3
            with tf.variable_scope('deconv_4'):
                outputs = tf.layers.conv2d_transpose(output, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.tanh(outputs, name='outputs')

        self.reuse = True
        # saving variables for regularization
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        return outputs