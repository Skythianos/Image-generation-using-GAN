import cv2
import os, sys
import tensorflow as tf
import random
import logging
import matplotlib.pyplot as ply
import math
import numpy as np


class DataManager():
    # init DataManager with Logger
    def __init__(self, dir="flowers", image_size=64, image_depth=3):
        self.path=os.getcwd()+ '/' + dir
        self.image_size=image_size
        self.image_depth=image_depth
        self.actual_batch = []
        self.logger = Logger("DataManager")
        self.logger.debug("Inited.")

    def get_batch(self, batch_size=128, augmentation=False):
        self.actual_batch=[]
        # reads file labels
        files = os.listdir(self.path)
        images = []
        for i in range(batch_size):
            #choosing random pictures
            choosen=files[i]
            self.logger.debug(choosen)
            #convert to 64x64x3 tensor
            images.append(tf.image.resize_images(cv2.imread(self.path+'/'+choosen),[64,64]))
        # data normalization, values will be between -1 and 1
        self.actual_batch = tf.reverse_v2(tf.subtract(tf.div(images, 127.5), 1.0), axis=[-1])

        return self.actual_batch


class Logger():
    def __init__(self,name, path='/log'):
        self.path = os.getcwd() + '/log/'
        self.logger = logging.getLogger(name)
        self.config_logger(self.logger)

    def config_logger(self, logger):
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.path + logger.name + '.log')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - [%(name)s] - [%(levelname)s]: %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
