from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pandas as pd
import numpy as np
import scipy.misc

from ops import *
from utils import *
import re

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=60, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_path="./data", training_subset=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.latent_dim = 60
    self.h_dim = 1000
    self.n_pixels = self.input_height*self.input_height

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.data_path = data_path
    
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    df = pd.read_csv(os.path.join(self.data_path,"scored.csv"))
    labeled_df = pd.read_csv(os.path.join(self.data_path,"labeled.csv"))
    labeled_df = labeled_df[labeled_df['Actual']==1.0]
    self.all_scores = np.array(df.sort_values(by='Id')['Actual'])
    self.data = glob(os.path.join(self.data_path, self.dataset_name, self.input_fname_pattern))
    raw_labeled_data = glob(os.path.join(self.data_path, "labeled", self.input_fname_pattern))
    self.training_subset = training_subset
    
    if not training_subset:
        training_subset = len(self.all_scores)

    self.all_scores = self.all_scores[:training_subset]
    self.data = self.data[:training_subset]
    self.labeled_data = []
#    print("len of labeled_data ", len(self.labeled_data))
    for d in raw_labeled_data:
      name = re.sub(".png", "", re.findall(r"\d*\.png", d)[0])
      if int(name) in list(labeled_df["Id"]):
        self.labeled_data.append(d)
    
#    print("labeled_data ", self.labeled_data[1], self.labeled_data[1][40:47])
#    print("len of labeled_data ", len(self.labeled_data))
#      print(len(self.all_scores))
#      print(os.path.join(self.data_path, self.dataset_name, self.input_fname_pattern), self.data[:10])
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()


  def build_model(self):
  
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    #if self.crop:
    image_dims = [self.output_height, self.output_width, self.c_dim]
    #else:
    #  image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.labeled_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='labeled_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
    self.scores = tf.placeholder(tf.float32, [self.batch_size], name='similarity_scores')
    
    inputs = self.inputs
    labeled_inputs = self.labeled_inputs
    scores = self.scores
    sample_inputs = self.sample_inputs

#    self.z = tf.placeholder(
#      tf.float32, [None, self.z_dim], name='z')

    # connecting generator and descriminator networks
    self.labeled_input_flattened = tf.reshape(labeled_inputs, (self.batch_size,self.input_height*self.input_width))
    self.z, self.logstd, self.mu= self.encoder(self.labeled_input_flattened)
    self.g_flattened = self.generator(self.z)
    self.D, self.D_logits, self.D_similarity = self.discriminator(inputs, scores)
    self.z_sum = histogram_summary("z", self.z)

#      self.sampler = self.sampler(self.z)
    self.G = tf.reshape(self.g_flattened, (self.batch_size, self.input_height, self.input_width, 1))
    self.D_, self.D_logits_, _ = self.discriminator(self.G, scores=tf.zeros_like(self.scores), reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    
    # defining loss functions

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
    # loss value for discriminator
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.zeros_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    # loss value for generator
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

    # loss value for VAE
    log_likelihood = tf.reduce_sum(self.labeled_input_flattened*tf.log(self.g_flattened + 1e-9)+(1 - self.labeled_input_flattened)*tf.log(1 - self.g_flattened + 1e-9), reduction_indices=1)
    KL_term = -.5 * tf.reduce_sum(1 + 2*self.logstd - tf.pow(self.mu,2) - tf.exp(2*self.logstd), reduction_indices=1)
    variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
    self.e_loss = -variational_lower_bound
    

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
    self.e_loss_sum = scalar_summary("e_loss", self.e_loss)

    # trainable variables
    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.eg_vars = [var for var in t_vars if 'g_' in var.name or 'e_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
  
  
  def predict(self, config):
    " function predict: predict similarity scores and d values for a subset of training images"
    
    df = pd.read_csv(os.path.join(self.data_path,"scored.csv"))
    self.all_scores = np.array(df.sort_values(by='Id')['Actual'])
    self.data = glob(os.path.join(self.data_path, self.dataset_name, self.input_fname_pattern))

    if not self.training_subset:
      print("provide prediction set")
    # getting the test subset
    self.test_subset = min(self.training_subset + self.batch_size, len(self.all_scores))

    self.all_scores = self.all_scores[self.training_subset:self.test_subset]
    self.data = self.data[self.training_subset:self.test_subset]
    batch_scores = self.all_scores

    self.D, self.D_logits, self.D_similarity = \
        self.discriminator(self.inputs, self.scores, reuse=True)

    batch_files = self.data
    batch = [
      get_image(batch_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop,
                grayscale=self.grayscale) for batch_file in batch_files]
    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

    self.d_sum_predict = histogram_summary("d_predicted", self.D)
    self.d_merge_sum_predict = merge_summary(
        [self.d_loss_real_sum, self.d_loss_sum, self.d_sum_predict])

    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
          .astype(np.float32)

    summary_str, D_similarity, D = self.sess.run([self.d_merge_sum_predict, self.D_similarity, self.D],
            feed_dict={ self.inputs: batch_images, self.z: batch_z, self.scores: batch_scores })
    
#    print("actual similarities: ", batch_scores)
#    print("similarities predicted: ", D_similarity)
#    print("D values:", D)

    self.predict_query(config)
    self.writer = SummaryWriter("./logs", self.sess.graph)
    self.writer.add_summary(summary_str, 0)



  def predict_query(self, config):
      " function predict_query: predict similarity scores for query images"
      self.data = glob(os.path.join(self.data_path, "query", self.input_fname_pattern))
      self.D, self.D_logits, self.D_similarity = \
              self.discriminator(self.inputs, self.scores, reuse=True)
      #print(self.data)
      batch_idxs = int(np.ceil(len(self.data)/config.batch_size))
      for idx in range(0, batch_idxs):
          start_idx = idx*config.batch_size
          end_idx = min(len(self.data), (idx+1)*config.batch_size)
          batch_files = self.data[start_idx:end_idx]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            
          if len(batch_images) < self.batch_size:
            size_diff = self.batch_size - len(batch_images)
            zeros = np.zeros(shape=(self.output_height, self.output_width, 1))
            for i in range(size_diff):
              batch_images = np.vstack((batch_images,[zeros]))
              batch_files = np.concatenate((batch_files,["dummy"]))

          self.d_sum_predict = histogram_summary("d_predicted", self.D)
          self.d_merge_sum_predict = merge_summary(
                [self.d_loss_real_sum, self.d_loss_sum, self.d_sum_predict])

          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                  .astype(np.float32)

          summary_str, D_similarity, D = self.sess.run([self.d_merge_sum_predict, self.D_similarity, self.D],
                    feed_dict={ self.inputs: batch_images, self.z: batch_z, self.scores: [0]*config.batch_size})
          for i in range(self.batch_size):
            print(batch_files[i], ",", D_similarity[i])


  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    e_optim = tf.train.AdadeltaOptimizer().minimize(self.e_loss, var_list=self.eg_vars)
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.e_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.e_loss_sum])
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    sample_files = self.data[0:self.sample_num]
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for sample_file in sample_files]
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      self.data = glob(os.path.join(
        self.data_path, config.dataset, self.input_fname_pattern))
#        batch_idxs = min(len(self.data), config.batch_size) // config.batch_size
      batch_idxs = min(len(self.all_scores), config.train_size) // config.batch_size
      for idx in xrange(0, batch_idxs):
        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        labeled_batch_files = self.labeled_data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_scores = self.all_scores[idx*config.batch_size:(idx+1)*config.batch_size].astype(np.float32)
        train_data = np.array(list(zip(batch_files, batch_scores)))

        random.shuffle(train_data)
        random.shuffle(labeled_batch_files)
        batch_files, batch_scores = zip(*train_data)
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        labeled_batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in labeled_batch_files]


        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          labeled_batch_images = np.array(labeled_batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)
          labeled_batch_images = np.array(labeled_batch).astype(np.float32)


        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.inputs: batch_images, self.scores: batch_scores , self.labeled_inputs: labeled_batch_images})
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str, reconstruction = self.sess.run([e_optim, self.e_sum, self.G],
          feed_dict={ self.inputs: batch_images, self.scores: batch_scores, self.labeled_inputs: labeled_batch_images})
        self.writer.add_summary(summary_str, counter)
        reconstruction_image = (np.reshape(reconstruction, (self.batch_size, self.output_width, self.output_height)))

        if counter % 10 == 0:
          scipy.misc.imsave("samples/{}.png".format(counter),reconstruction_image[0])

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.inputs: batch_images, self.scores: batch_scores, self.labeled_inputs: labeled_batch_images})
        self.writer.add_summary(summary_str, counter)
        
        errD_fake = self.d_loss_fake.eval({ self.inputs: batch_images, self.scores: batch_scores , self.labeled_inputs: labeled_batch_images})
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.scores: batch_scores , self.labeled_inputs: labeled_batch_images})
        errG = self.g_loss.eval({ self.inputs: batch_images, self.scores: batch_scores , self.labeled_inputs: labeled_batch_images})
        errE = self.e_loss.eval({ self.inputs: batch_images, self.scores: batch_scores , self.labeled_inputs: labeled_batch_images})
       
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, errE))

        '''if np.mod(counter, 100) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.d_loss, self.g_loss],
              feed_dict={
                  self.inputs: sample_inputs,
                  self.scores: batch_scores,
              },
            )
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          except:
            print("one pic error!...")'''

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)


  def discriminator(self, image, scores, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
#        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
#        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h4_lin')
      h5 = tf.pow(tf.subtract(h4, scores), tf.fill(tf.shape(h4), 2.0))
      print("shape of discriminator output(h4): ", tf.shape(h4))

      return tf.nn.sigmoid(h5), h5, h4

  def FC_layer(self, X, W, b):
    return tf.matmul(X, W) + b
  
  
  def weight_variable(self, shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


  def bias_variable(self, shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

  
  def encoder(self, X):
    with tf.variable_scope("generator") as scope:

      W_enc = self.weight_variable([self.n_pixels, self.h_dim], 'e_W_enc')
      b_enc = self.bias_variable([self.h_dim], 'e_b_enc')

      # tanh activation function to replicate original model

      h_enc = tf.nn.tanh(self.FC_layer(X, W_enc, b_enc))

      W_mu = self.weight_variable([self.h_dim, self.latent_dim], 'e_W_mu')
      b_mu = self.bias_variable([self.latent_dim], 'e_b_mu')
      mu = self.FC_layer(h_enc, W_mu, b_mu)

      W_logstd = self.weight_variable([self.h_dim, self.latent_dim], 'e_W_logstd')
      b_logstd = self.bias_variable([self.latent_dim], 'e_b_logstd')
      logstd = self.FC_layer(h_enc, W_logstd, b_logstd)

      # Reparameterization trick

      noise = tf.random_normal([1, self.latent_dim])
      z = mu + tf.multiply(noise, tf.exp(.5*logstd))
      return z, logstd, mu
  
  def generator(self, z):
  
    with tf.variable_scope("generator") as scope:

      W_dec = self.weight_variable([self.latent_dim, self.h_dim], 'g_W_dec')
      b_dec = self.bias_variable([self.h_dim], 'g_b_dec')
      h_dec = tf.nn.tanh(self.FC_layer(z, W_dec, b_dec))

      W_reconstruct = self.weight_variable([self.h_dim, self.n_pixels], 'g_W_reconstruct')
      b_reconstruct = self.bias_variable([self.n_pixels], 'g_b_reconstruct')
      reconstruction = tf.nn.sigmoid(self.FC_layer(h_dec, W_reconstruct, b_reconstruct))
      return reconstruction


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
