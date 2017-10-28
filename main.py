import tensorflow as tf
import time

from DCGAN import DCGAN
from DataManager import DataManager
from DataManager import Logger

path='./flowers/'
# You can download data from: http://www.robots.ox.ac.uk/~vgg/data/flowers/
# Citation: M-E. Nilsback and A. Zisserman. Automated flower classification over a large number
# of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and
# Image Processing, Dec 2008.

dcgan = DCGAN()
data_man = DataManager()
logger = Logger('Main')

train_image = data_man.get_batch(batch_size=32,augmentation=True)

images = dcgan.sample_images()
losses = dcgan.loss(train_image)

# adding the regularizator
graph = tf.get_default_graph()
features_g = tf.reduce_mean(graph.get_tensor_by_name('g/D/conv_4/outputs:0'), 0)
features_t = tf.reduce_mean(graph.get_tensor_by_name('t/D/conv_4/outputs:0'), 0)
losses[dcgan.G] += tf.multiply(tf.nn.l2_loss(features_g - features_t), 0.05)
start_time = time.time()

# creating variables for tensorboard
tf.summary.scalar('g_loss', losses[dcgan.G])
tf.summary.scalar('d_loss', losses[dcgan.D])

train_op = dcgan.train(losses, learning_rate=0.0001)

summary_op = tf.summary.merge_all()

# config for GPU usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# with tf.Session(config=config) as sess: # sometimes dont work
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logger.path,graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners(sess=sess)

    for step in range(10001):
        _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.G], losses[dcgan.D]])
        # in every 50 epoch saving the loss values
        if step%50==0:
            print("Epoch: " + str(step) + " Loss: G(" + str(g_loss_value) + "), D(" + str(d_loss_value) + ")")
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        # every 100 epoch saving the generated images
        if step%100 == 0 :
            generated = sess.run(images)
            with open('pic'+str(step), 'wb') as f:
                f.write(generated)

    # prints the duration of training, also can finde on tensorboard graphs
    print("Duration: " + str((time.time() - start_time)/60) + " mins")
