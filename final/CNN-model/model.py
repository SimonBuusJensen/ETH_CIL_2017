import logging
import pickle as pkl
import time

from utils import *


class Model:
    def __init__(self, sess):

        # Weight & Bias of the model
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable(name='wc1', shape=[5, 5, 1, 32], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable(name='wc2', shape=[5, 5, 32, 64], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # fully connected layer. Taking into account the fact that conv layers and max pool down scales the image.
            # That is the reason for the "-4" and "/ 2". A 5X5 kernel reduces the size by 4 and a max pool reduces the
            # size of the image to half the size.
            'wd1': tf.get_variable(name='wd1',
                                   shape=[
                                       ((((cfg['down_scale_img_h'] - 4) / 2) - 4) / 2) *
                                       ((((cfg['down_scale_img_h'] - 4) / 2) - 4) / 2) * 64, 100],
                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable(name='out', shape=[100, 1], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'bc1': tf.get_variable(name='bc1', shape=[32], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable(name='bc2', shape=[64], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable(name='bd1', shape=[100], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.Variable(tf.zeros([1], dtype=tf.float32))
        }

        if cfg['mode'] == 'train':
            # Define model name based on current date and time. For logging, saving checkpoint and training outputs
            self.model_name = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()).replace(" ", "")
            self.make_paths()
            self.make_project_dependent_dirs()
            self.make_model_name_dirs()

            # Define logger for logging output
            logging.basicConfig(filename=os.path.join(cfg['path']['log_dir'], self.model_name) + "/info_log.txt",
                                level=logging.INFO)
            self.logger = logging.getLogger(os.path.join(cfg['path']['log_dir'], self.model_name) + "/info_log.txt")

            print_and_log("initializing model...", self.logger)
            self.build_model()
            self.sess = sess

            # Output file for writing training
            self.training_and_test_output_file = open(self.training_and_test_output_path, 'a')
            self.training_and_test_output_file.write("Epoch:,mean_train_loss:,mean_test_loss:\n")

        else:
            pass

    def make_paths(self):
        self.output_dir = cfg['path']['output_dir']
        self.output_model_dir = os.path.join(self.output_dir, self.model_name)
        self.training_and_test_output_path = \
            os.path.join(self.output_dir, self.model_name, cfg['path']['train_output_file'])

        self.checkpoint_dir = cfg['path']['checkpoint_dir']
        self.checkpoint_model_dir = os.path.join(self.checkpoint_dir, self.model_name)
        self.checkpoint_file_name = cfg['path']['checkpoint_file']

        self.log_dir = cfg['path']['log_dir']
        self.log_model_dir = os.path.join(self.log_dir, self.model_name)

        self.matrices_dir = os.path.join(cfg['path']['data_dir'], cfg['path']['matrices_dir'])

        self.scored_img_dir = os.path.join(cfg['path']['data_dir'], cfg['path']['scored_img_dir'])
        self.scored_img_csv = os.path.join(cfg['path']['data_dir'], cfg['path']['scored_img_csv'])
        self.query_img_dir = os.path.join(cfg['path']['data_dir'], cfg['path']['query_img_dir'])
        self.query_img_csv = os.path.join(cfg['path']['data_dir'], cfg['path']['query_img_csv'])

    # Make project directories, if they don't already exist
    def make_project_dependent_dirs(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.matrices_dir):
            os.mkdir(self.matrices_dir)

    # Make a directory in the output dir, log dir and checkpoint dir to store results of current run
    def make_model_name_dirs(self):
        os.mkdir(self.output_model_dir)
        os.mkdir(self.checkpoint_model_dir)
        os.mkdir(self.log_model_dir)

    # Create the CNN network
    def convolutional_network(self, input_image):
        # Convolution Layer 1
        conv1_layer = tf.nn.conv2d(input=input_image, filter=self.weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1_layer = tf.nn.bias_add(conv1_layer, self.biases['bc1'])
        conv1_layer = tf.nn.relu(conv1_layer)

        # Max Pooling 1
        max1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # Convolution Layer 2
        conv2_layer = tf.nn.conv2d(input=max1_layer, filter=self.weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_layer = tf.nn.bias_add(conv2_layer, self.biases['bc2'])
        conv2_layer = tf.nn.relu(conv2_layer)

        # Max Pooling 2
        max2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1_layer = tf.reshape(max2_layer, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1_layer = tf.add(tf.matmul(fc1_layer, self.weights['wd1']), self.biases['bd1'])

        # Output, regression prediction
        out_layer = tf.add(tf.matmul(fc1_layer, self.weights['out']), self.biases['out'])
        return out_layer

    def build_model(self):
        print_and_log("", self.logger)
        print_and_log("building model...", self.logger)

        # print and set desired size of input images
        print_and_log("Down scaling enabled is: %r" % (cfg['enabled_down_scaling']), self.logger)
        if cfg['enabled_down_scaling']:
            self.image_dims = [cfg['down_scale_img_h'], cfg['down_scale_img_w'], 1]
            print_and_log("Image height: %.f Image width: %.f" % (cfg['down_scale_img_h'], cfg['down_scale_img_w']),
                          self.logger)
        else:
            self.image_dims = [cfg['img_h'], cfg['img_w'], 1]
            print_and_log("Image height: %.f Image width: %.f" % (cfg['img_h'], cfg['img_w']), self.logger)

        # Placeholders for input images and corresponding labels
        self.X = tf.placeholder(tf.float32, [cfg['batch_size']] + self.image_dims, name='input_images')
        self.Y = tf.placeholder(tf.float32, shape=[cfg['batch_size'], 1], name='label')

        # self.discriminator_logits = self.discriminator(self.inputs, reuse=False)
        self.x_logits = self.convolutional_network(self.X)
        self.loss_function = tf.reduce_sum(tf.abs(tf.subtract(x=self.x_logits, y=self.Y)))

        self.saver = tf.train.Saver()

    # Get the train, test and query data as numpy matrices instead of images.
    def get_data(self):

        # Dividing scored data into train and test set
        scored_img_dict = csv_to_dict(self.scored_img_csv)

        scored_image_indexes = list(scored_img_dict.keys())
        train_image_indexes = scored_image_indexes[:int(len(scored_image_indexes) * 0.9)]
        test_image_indexes = scored_image_indexes[int(len(scored_image_indexes) * 0.9):]

        print_and_log("Number of training samples: %.f" % int(len(train_image_indexes)), self.logger)
        print_and_log("Number of test samples: %.f" % int(len(test_image_indexes)), self.logger)
        print_and_log("", self.logger)

        x_train_file = os.path.join(self.matrices_dir,
                                    ("x_train" + str(self.image_dims[0]) + "-" + str(self.image_dims[1]) + ".pkl"))
        y_train_file = os.path.join(self.matrices_dir,
                                    ("y_train" + str(self.image_dims[0]) + "-" + str(self.image_dims[1]) + ".pkl"))

        if not os.path.exists(x_train_file) or not os.path.exists(y_train_file):
            print_and_log("Converting train images to matrices...", self.logger)
            x_train, y_train = image2matrix(self.scored_img_dir, train_image_indexes, scored_img_dict)
            print_and_log(("Dumping the matrices to: " + self.matrices_dir + " for future usage..."), self.logger)
            pkl.dump(x_train, open(x_train_file, "wb"))
            pkl.dump(y_train, open(y_train_file, "wb"))
        else:
            print_and_log(("Loading training images from " + self.matrices_dir + "..."), self.logger)
            x_train = pkl.load(open(x_train_file, "rb"))
            y_train = pkl.load(open(y_train_file, "rb"))

        x_test_file = os.path.join(self.matrices_dir,
                                   ("x_test" + str(self.image_dims[0]) + "-" + str(self.image_dims[1]) + ".pkl"))
        y_test_file = os.path.join(self.matrices_dir,
                                   ("y_test" + str(self.image_dims[0]) + "-" + str(self.image_dims[1]) + ".pkl"))

        if not os.path.exists(x_test_file) or not os.path.exists(y_test_file):
            print_and_log("Converting test images to matrices...", self.logger)
            x_test, y_test = image2matrix(self.scored_img_dir, test_image_indexes, scored_img_dict)
            print_and_log(("Dumping the matrices to: " + self.matrices_dir + " for future usage..."), self.logger)
            pkl.dump(x_test, open(x_test_file, "wb"))
            pkl.dump(y_test, open(y_test_file, "wb"))
        else:
            print_and_log(("Loading test images from " + self.matrices_dir + "..."), self.logger)
            x_test = pkl.load(open(x_test_file, "rb"))
            y_test = pkl.load(open(y_test_file, "rb"))

        # Preparing query images for outputting prediction
        self.query_image_indexes = pd.read_csv(self.query_img_csv)['Id'].values
        x_query_file = os.path.join(self.matrices_dir,
                                    ("x_query" + str(self.image_dims[0]) + "-" + str(self.image_dims[1]) + ".pkl"))

        if not os.path.exists(x_query_file):
            print_and_log("Converting query images to matrices...", self.logger)
            x_query = image2matrix(self.query_img_dir, self.query_image_indexes, scored_img_dict, query_file=True)
            print_and_log(("Dumping the matrices to: " + self.matrices_dir + " for future usage..."), self.logger)
            pkl.dump(x_query, open(x_query_file, "wb"))
        else:
            print_and_log(("Loading query images from " + self.matrices_dir + "..."), self.logger)
            x_query = pkl.load(open(x_query_file, "rb"))

        return x_train, y_train, x_test, y_test, x_query

    # Training function.
    def train(self, x_train, y_train, x_test, y_test, x_query):
        print_and_log("", self.logger)
        print_and_log("training model...", self.logger)
        print_and_log("Batch size: %.f" % cfg['batch_size'], self.logger)

        # Defining optimizer
        d_optimizer = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['adam_momentum']).minimize(self.loss_function)

        n_train_batches = (int(len(y_train)) / cfg['batch_size'])
        n_test_batches = (int(len(y_test)) / cfg['batch_size'])
        n_query_batches = int(len(x_query) / cfg['batch_size'])
        batch_display_step = cfg['batch_size'] * 10

        self.sess.run(tf.global_variables_initializer())
        global_counter = 1
        first_epoch = True

        for epoch in range(cfg['epochs']):

            epoch_start_time = time.time()
            epoch_loss = 0

            # Shuffling train data from epoch to epoch to get different batches every time
            print_and_log("Shuffling train data...", self.logger)
            permutation = np.random.permutation(x_train.shape[0])
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            print_and_log("done!", self.logger)

            for i in range(n_train_batches):

                x_train_batch = x_train[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]
                y_train_batch = y_train[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]

                train_loss, _ = self.sess.run([self.loss_function, d_optimizer],
                                              feed_dict={self.X: x_train_batch, self.Y: y_train_batch})

                if (i + 1) % (cfg['batch_size'] * 10) == 0:
                    print_and_log("Batch: %.f/%.f. Mean train batch loss: %.4f"
                                  % ((i + 1) % n_train_batches, n_train_batches, train_loss / cfg['batch_size']),
                                  self.logger)

                epoch_loss += train_loss
                global_counter += 1

            # Print the results from the current epoch
            print_and_log("Epoch %.f/%.f, time: %.2f, total epoch loss: %.6f"
                          % (epoch + 1, cfg['epochs'], time.time() - epoch_start_time, epoch_loss), self.logger)

            ###################################################################################
            # Test how well the model predicts scores on the test set, to check if it over fits
            ###################################################################################
            if epoch % cfg['test_step'] == 0 and not first_epoch:
                print_and_log("testing...", self.logger)

                total_test_loss = 0

                for i in range(n_test_batches):
                    x_test_batch = x_test[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]
                    y_test_batch = y_test[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]

                    test_loss = self.sess.run([self.loss_function],
                                              feed_dict={self.X: x_test_batch, self.Y: y_test_batch})

                    total_test_loss += test_loss[0]

                    if (i + 1) % batch_display_step == 0:
                        print_and_log("Batch: %.f/%.f. Mean test batch loss: %.4f"
                                      % ((i + 1) % n_train_batches, n_train_batches, test_loss / cfg['batch_size']),
                                      self.logger)

                self.training_and_test_output_file = open(self.training_and_test_output_path, mode='a')
                self.training_and_test_output_file.write("%.f, %.4f, %.4f\n"
                                                         % (epoch,
                                                            ((epoch_loss / n_train_batches) / cfg['batch_size']),
                                                            ((total_test_loss / n_test_batches) / cfg['batch_size'])))
                self.training_and_test_output_file.close()

                print_and_log("Mean test loss: %.4f" % ((total_test_loss / n_test_batches) / cfg['batch_size']),
                              self.logger)

            ###################################################################################
            # Predict scores for the query images and save result to output folder.
            ###################################################################################

            if epoch % cfg['predict_step'] == 0 and not first_epoch:

                query_img_logits = []

                for i in range(n_query_batches):
                    x_query_batch = x_query[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]
                    logits = self.sess.run([self.x_logits], feed_dict={self.X: x_query_batch})
                    query_img_logits.append(logits)

                # Export the predictions of the query examples to the output folder
                export_result_to_csv(query_img_logits, self.query_img_csv, self.model_name, epoch)

            # Save the network model to the checkpoint directory every x epoch
            if epoch % cfg['save_step'] == 0 and not first_epoch:
                self.saver.save(sess=self.sess,
                                save_path=os.path.join(self.checkpoint_model_dir, self.checkpoint_file_name),
                                global_step=global_counter)

            first_epoch = False
