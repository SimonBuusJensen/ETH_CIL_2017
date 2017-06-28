import logging
import os
import time
import random

import pandas as pd
import tensorflow as tf
import numpy as np
import pickle as pkl

from config import cfg
from data_reader import csv_to_dict
from utils import BatchNorm
from utils import DCGAN_conv2d
from utils import create_mini_batches
from utils import export_result_to_csv
from utils import image2matrix
from utils import image2matrix_query_file
from utils import linear
from utils import print_and_log


class DCGANModel:
    def __init__(self, sess):
        self.sess = sess

        if cfg['mode'] == "train":
            self.time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()).replace(" ", "")
            os.mkdir(os.path.join(cfg['path']['checkpoint_dir'], self.time_str))
            self.model_name = self.time_str

            if not os.path.exists(os.path.join(cfg['path']['log_dir'], self.model_name)):
                os.mkdir(os.path.join(cfg['path']['log_dir'], self.model_name))

            logging.basicConfig(filename=os.path.join(cfg['path']['log_dir'], self.model_name) + "/info_log.txt",
                                level=logging.INFO)
            self.logger = logging.getLogger(os.path.join(cfg['path']['log_dir'], self.model_name) + "/info_log.txt")

            if not os.path.exists(cfg['path']['output_dir']):
                os.mkdir(cfg['path']['output_dir'])

            if not os.path.exists(cfg['path']['checkpoint_dir']):
                os.mkdir(cfg['path']['checkpoint_dir'])

            if not os.path.exists(cfg['path']['log_dir']):
                os.mkdir(cfg['path']['log_dir'])

            print_and_log("initializing model...", self.logger)
            self.build_model()

        print_and_log("Cropping enabled is: %r" % (cfg['enabled_cropping']), self.logger)
        if cfg['enabled_cropping']:
            print_and_log("Image height: %.f Image width: %.f" % (cfg['crop_img_h'], cfg['crop_img_w']), self.logger)
        else:
            print_and_log("Image height: %.f Image width: %.f" % (cfg['img_h'], cfg['img_w']), self.logger)

    @staticmethod
    def discriminator(image, reuse=False):
        discriminator_filter_dimension = 16
        batch_normalization_1 = BatchNorm(name='d_bn1')
        batch_normalization_2 = BatchNorm(name='d_bn2')
        batch_normalization_3 = BatchNorm(name='d_bn3')

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = tf.nn.relu(DCGAN_conv2d(image, discriminator_filter_dimension, name='d_h0_conv'))
            h1 = tf.nn.relu(
                batch_normalization_1(DCGAN_conv2d(h0, discriminator_filter_dimension * 2, name='d_h1_conv')))
            h2 = tf.nn.relu(
                batch_normalization_2(DCGAN_conv2d(h1, discriminator_filter_dimension * 4, name='d_h2_conv')))
            h3 = tf.nn.relu(
                batch_normalization_3(DCGAN_conv2d(h2, discriminator_filter_dimension * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [cfg['batch_size'], -1]), 1, 'd_h4_lin')

            return h4

    def build_model(self):
        print_and_log("", self.logger)
        print_and_log("building model...", self.logger)

        # Down scale the image if cropping is enabled. Will decrease accuracy but increase training speed
        if cfg['enabled_cropping']:
            image_dims = [cfg['crop_img_h'], cfg['crop_img_w'], 1]
        else:
            image_dims = [cfg['img_h'], cfg['img_w'], 1]

        self.inputs = tf.placeholder(tf.float32, [cfg['batch_size']] + image_dims, name='input_images')
        self.labels = tf.placeholder(tf.float32, shape=[cfg['batch_size'], 1], name='label')

        self.discriminator_logits = self.discriminator(self.inputs, reuse=False)

        self.discriminator_loss = tf.reduce_mean(tf.abs(tf.subtract(x=self.discriminator_logits, y=self.labels)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.saver = tf.train.Saver()

    def train(self):
        print_and_log("", self.logger)
        print_and_log("training model...", self.logger)
        d_optimizer = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['adam_momentum']). \
            minimize(self.discriminator_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()

        global_counter = 1

        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # if could_load:
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        # Preparing scored images and dividing into train and test set
        score_file = cfg['path']['scored_data_file_path']
        scored_img_dict = csv_to_dict(score_file)
        prefixes_scored_data = list(scored_img_dict.keys())
        train_scored_data = prefixes_scored_data[:int(len(prefixes_scored_data) * 0.8)]
        test_score_data = prefixes_scored_data[int(len(prefixes_scored_data) * 0.8):]

        # Preparing query images for outputting prediction
        sample_file = cfg['path']['query_data_file_path']
        sample_data_frame = pd.read_csv(sample_file)
        os.mkdir(os.path.join(cfg['path']['output_dir'], self.model_name))
        training_output_file = open(os.path.join(cfg['path']['output_dir'], self.model_name) + "/train_output.csv", 'a')
        training_output_file.write("Epoch:,mean_train_loss:,mean_test_loss:\n")

        print_and_log("Number of training samples: %.f" % int(len(train_scored_data)), self.logger)
        print_and_log("Batch size: %.f" % cfg['batch_size'], self.logger)
        print_and_log("Number of batches per epoch: %.f" % (int(len(train_scored_data)) / cfg['batch_size']),
                      self.logger)

        print_and_log("", self.logger)

        print_and_log("Converting train images to matrices...", self.logger)
        x_train_matrix = pkl.load(open("data/matrices/x_train.pkl", "rb"))
        y_train_labels = pkl.load(open("data/matrices/y_train.pkl", "rb"))

        # x_train_matrix, y_train_labels = image2matrix(cfg['path']['scored_data_path'],
        #                                               train_scored_data, scored_img_dict)
        # pkl.dump(x_train_matrix, open("data/matrices/x_train.pkl", "wb"))
        # pkl.dump(y_train_labels, open("data/matrices/y_train.pkl", "wb"))

        print_and_log("Converting test images to matrices...", self.logger)
        x_test_matrix = pkl.load(open("data/matrices/x_test.pkl", "rb"))
        y_test_labels = pkl.load(open("data/matrices/y_test.pkl", "rb"))
        # x_test_matrix, y_test_labels = image2matrix(cfg['path']['scored_data_path'],
        #                                             test_score_data, scored_img_dict)
        # pkl.dump(x_test_matrix, open("data/matrices/x_test.pkl", "wb"))
        # pkl.dump(y_test_labels, open("data/matrices/y_test.pkl", "wb"))

        print_and_log("Converting query images to matrices...", self.logger)
        x_query_matrix = pkl.load(open("data/matrices/x_query.pkl", "rb"))
        # x_query_matrix = image2matrix_query_file(cfg['path']['query_data_path'], sample_data_frame['Id'].values)
        # pkl.dump(x_query_matrix, open("data/matrices/x_query.pkl", "wb"))


        first_epoch = True
        for epoch in range(cfg['epochs']):

            epoch_start_time = time.time()
            epoch_loss = 0
            n_train_batches = (int(len(train_scored_data)) / cfg['batch_size'])
            n_test_batches = (int(len(test_score_data)) / cfg['batch_size'])

            print_and_log("Shuffling train data", self.logger)
            train_data = np.array(zip(x_train_matrix, y_train_labels))
            random.shuffle(train_data)
            x_train_matrix, y_train_labels = zip(*train_data)
            print_and_log("done", self.logger)

            # train_mini_batches = create_mini_batches(cfg['batch_size'], train_scored_data)
            # for idx, batch in enumerate(train_mini_batches):

            for idx in range(n_train_batches):

                # x_batch, y_batch = image2matrix(cfg['path']['scored_data_path'], batch, scored_img_dict)
                x_batch = x_train_matrix[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                y_batch = y_train_labels[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]

                # Update Discriminator network
                d_loss, _ = self.sess.run([self.discriminator_loss, d_optimizer],
                                          feed_dict={self.inputs: x_batch, self.labels: y_batch})

                if (idx + 1) % 100 == 0:
                    print_and_log("Batch: %.f/%.f. Batch loss: %.4f"
                                  % ((idx + 1) % n_train_batches, n_train_batches, d_loss), self.logger)

                global_counter += 1
                epoch_loss += d_loss

            # Test how well the model predict the scores given the test set.
            if epoch % 2 == 0 and not first_epoch:
                print_and_log("testing...", self.logger)

                total_test_accuracy = 0
                for idx in range(n_test_batches):
                    x_test_batch = x_test_matrix[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                    y_test_batch = y_test_labels[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]

                    d_test_accuracy = self.sess.run([self.discriminator_loss],
                                                    feed_dict={self.inputs: x_test_batch, self.labels: y_test_batch})
                    total_test_accuracy += d_test_accuracy[0]

                training_output_file.write("%.f, %.4f, %.4f\n"
                                           % (epoch,
                                              (epoch_loss / n_train_batches),
                                              (total_test_accuracy / n_test_batches)))

                print_and_log("Mean test accuracy: %.2f" % (total_test_accuracy / n_test_batches), self.logger)

            print_and_log("Epoch %.f/%.f, time: %.2f, epoch_loss: %.6f"
                          % (epoch + 1, cfg['epochs'], time.time() - epoch_start_time, epoch_loss), self.logger)

            # Predict scores for query examples and save result to output folder.
            ids = sample_data_frame['Id'].values
            predictions = []
            n_query_batches = int(len(ids) / cfg['batch_size'])
            # The query images are split into batch sized batches so that it is possible to feed it to the discriminator
            for idx in range(n_query_batches):

                x_query_batch = x_query_matrix[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                pred = self.discriminator_logits.eval(feed_dict={self.inputs: x_query_batch}, session=self.sess)
                predictions.append(pred)

            # Export the predictions of the query examples to the output folder
            export_result_to_csv(ids, predictions, sample_data_frame.columns.values, self.time_str, epoch)

            # Save the network model to the checkpoint directory every x epoch
            if epoch % 10 == 0 and not first_epoch:
                self.saver.save(sess=self.sess,
                                save_path=os.path.join(cfg['path']['checkpoint_dir'], self.model_name + "/"),
                                global_step=global_counter)

            first_epoch = False

        training_output_file.close()
