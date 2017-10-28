import tensorflow as tf
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    # bulding both models and the random input
    def __init__(self, batch_size=32, s_size=4, z_dim=100):
        self.batch_size = batch_size
        self.G = Generator(s_size=s_size)
        self.D = Discriminator()
        self.z = tf.random_uniform([self.batch_size, z_dim], minval=-0.9, maxval=0.9)

    # setting up de losses
    def loss(self, traindata):
        generated = self.G(self.z, training=True)
        gen_outputs = self.D(generated, training=True, name='g')
        true_outputs = self.D(traindata, training=True, name='t')

        tf.add_to_collection('g_loss',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=gen_outputs)))
        # Discriminator have 2 type of losses, one is for fake data and one is for real
        tf.add_to_collection('d_loss',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),  # ones for real
                    logits=true_outputs)))

        tf.add_to_collection('d_loss',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64), # zeros for false
                    logits=gen_outputs)))

        return {
            self.G: tf.add_n(tf.get_collection('g_loss'), name='total_g_loss'),
            self.D: tf.add_n(tf.get_collection('d_loss'), name='total_d_loss'),
        }

    # minimizeing loss with the Optimizer
    def train(self, losses, learning_rate=0.003, beta1=0.9):

        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        g_opt_op = g_opt.minimize(losses[self.G], var_list=self.G.variables)
        d_opt_op = d_opt.minimize(losses[self.D], var_list=self.D.variables)

        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    # pretraining the generator
    def g_pre_train(self, losses, learning_rate=0.003, beta1=0.9):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.G], var_list=self.G.variables)

        with tf.control_dependencies([g_opt_op]):
            return tf.no_op(name='g_pre_train')

    # pretraining the discriminator
    def d_pre_train(self, losses, learning_rate=0.003, beta1=0.9):
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt_op = d_opt.minimize(losses[self.D], var_list=self.D.variables)

        with tf.control_dependencies([d_opt_op]):
            return tf.no_op(name='d_pre_train')


    def sample_images(self, row=1, col=1, inputs=None):
        # if inputs is None, we generate images with the training z
        if inputs is None:
            inputs = self.z
        # creating a batch of images and encoding to jpeg
        images = self.G(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]),"rgb")