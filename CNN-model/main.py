import tensorflow as tf
from model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        cnn_model = Model(sess)
        x_train, y_train, x_test, y_test, x_query = cnn_model.get_data()
        cnn_model.train(x_train, y_train, x_test, y_test, x_query)


if __name__ == "__main__":
    main()
