
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import sys
#get_ipython().magic(u'matplotlib inline')

np.random.seed(0)
tf.set_random_seed(0)


# In[2]:

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


# In[3]:

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # haha
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


# In[4]:

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, 
                 batch_size=100, 
                 train_keep_prob=1.0, 
                 warmup=False, 
                 batch_norm = True,
                iw=False):
        
        
        
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.importance_weighting = iw
        
        self.warmup = warmup
        self.batch_norm = batch_norm
        self.train_keep_prob = train_keep_prob
        
        self.epoch = 0.
        self.beta = 1.
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Keep Probability
        self.keep_prob = tf.placeholder(tf.float32)
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        #self._create_loss_optimizer()
        
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(-self._evidence_lower_bound(                    importance_weighting=self.importance_weighting))
        
        self.cost = -self._evidence_lower_bound(                    importance_weighting=self.importance_weighting)

        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)
        
        if self.batch_norm:
            self.scale1=tf.Variable(tf.ones([self.network_architecture['n_hidden_recog_1']]))
            self.bias1=tf.Variable(tf.zeros([self.network_architecture['n_hidden_recog_1']]))
            self.scale2=tf.Variable(tf.ones([self.network_architecture['n_hidden_recog_2']]))
            self.bias2=tf.Variable(tf.zeros([self.network_architecture['n_hidden_recog_2']]))

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq =             self._recognition_network(self.x,self.network_weights["weights_recog"], 
                                      self.network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

#         # Use generator to determine mean of
#         # Bernoulli distribution of reconstructed input
#         self.x_reconstr_mean = \
#             self._generator_network(network_weights["weights_gener"],
#                                     network_weights["biases_gener"])
            
            # Use generator to determine mean and (log) variance of
        # Gaussian distribution of reconstructed input
        self.x_reconstr_mean, self.x_reconstr_log_sigma_sq =             self._generator_network(self.z,self.network_weights["weights_gener"],
                                   self.network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, x, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        
        if self.batch_norm:
            layer_1 = tf.matmul(x, weights['h1'])
            mom_mean1, mom_var1 = tf.nn.moments(layer_1, [0])
            layer_1 = tf.nn.batch_normalization(layer_1,mom_mean1,mom_var1,self.scale1,self.bias1,1E-5)
            layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
            layer_1 = self.transfer_fct(layer_1)
        else:
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
            layer_1 = self.transfer_fct(layer_1) 
                  
        if self.batch_norm:
            layer_2 = tf.matmul(layer_1, weights['h2'])
            mom_mean2, mom_var2 = tf.nn.moments(layer_2, [0])
            layer_2 = tf.nn.batch_normalization(layer_2,mom_mean2,mom_var2,self.scale2,self.bias2,1E-5)
            layer_2 = tf.nn.dropout(layer_2, self.keep_prob)
            layer_2= self.transfer_fct(layer_2)
            
        else:
            layer_2 =tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])

            layer_2 = tf.nn.dropout(layer_2, self.keep_prob)
            
            layer_2= self.transfer_fct(layer_2)
        
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq =             tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, z, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) 
        
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        
        x_reconstr_mean =             tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        x_reconstr_log_sigma_sq =             0.5*tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                                  biases['out_log_sigma']))
            
        return (x_reconstr_mean, x_reconstr_log_sigma_sq)
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.\n",
        # Adding 1e-10 to avoid evaluatio of log(0.0)\n",
        reconstr_loss =             -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
            + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
            1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(self.beta * (1 + self.z_log_sigma_sq
            - tf.square(self.z_mean)
            - tf.exp(self.z_log_sigma_sq)), 1)
#         elif self.gen_distribution == 'gaussian':
#             #RECONSTRUCTION LOSS GAUSSIAN
#             reconstr_loss = \
#                 -tf.reduce_sum(-(0.5 * np.log(2 * np.pi)
#                 + self.x_reconstr_log_sigma_sq)
#                 - 0.5 * tf.square((self.x - self.x_reconstr_mean)
#                 / tf.exp(self.x_reconstr_log_sigma_sq)),
#                 1)
#             #LATENT LOSS GAUSSIAN
#             latent_loss = -0.5 * tf.reduce_sum(1 + 2*self.z_log_sigma_sq
#                 - tf.square(self.z_mean)
#                 - tf.exp(2*self.z_log_sigma_sq), 1)
        # average over batch
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        # Use ADAM optimizer
        self.optimizer =             tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            
    
    def _evidence_lower_bound(self,
                              monte_carlo_samples=5,
                              importance_weighting=False,
                              tol=1e-5):
        """
            Variational objective function
            ELBO = E(log joint log-likelihood) - E(log q)
                 = MC estimate of log joint - Entropy(q)
        """


        x_resampled = tf.tile(self.x, tf.constant([monte_carlo_samples, 1]))

        # Forward pass of data into latent space
        mean_encoder, log_variance_encoder = self._recognition_network(x_resampled,self.network_weights["weights_recog"], 
                                      self.network_weights["biases_recog"])

        random_noise = tf.random_normal(
            (self.batch_size * monte_carlo_samples, self.network_architecture['n_z']), 0, 1, dtype=tf.float32)

        # Reparameterization trick of re-scaling/transforming random error
        std_dev = tf.sqrt(tf.exp(log_variance_encoder))
        z = mean_encoder + std_dev * random_noise

        # Reconstruction/decoding of latent space
        mean_decoder, _ = self._generator_network(z,self.network_weights["weights_gener"],
                                   self.network_weights["biases_gener"])

        # Bernoulli log-likelihood reconstruction
        # TODO: other distributon types
        def bernoulli_log_joint(x):
            return tf.reduce_sum(
                (x * tf.log(tol + mean_decoder))
                    + ((1 - x) * tf.log(tol + 1 - mean_decoder)), 
                1)

        log2pi = tf.log(2.0 * np.pi)

        def gaussian_likelihood(data, mean, log_variance):
            """Log-likelihood of data given ~ N(mean, exp(log_variance))

            Parameters
            ----------
            data : 
                Samples from Gaussian centered at mean
            mean : 
                Mean of the Gaussian distribution
            log_variance : 
                Log variance of the Gaussian distribution
            Returns
            -------
            log_likelihood : float
            """

            num_components = data.get_shape().as_list()[1]
            variance = tf.exp(log_variance)
            log_likelihood = (
                -(log2pi * (num_components / 2.0))
                - tf.reduce_sum(
                    (tf.square(data - mean) / (2 * variance)) + (log_variance / 2.0),
                    1)
            )

            return log_likelihood

        def standard_gaussian_likelihood(data):
            """Log-likelihood of data given ~ N(0, 1)
            Parameters
            ----------
            data : 
                Samples from Guassian centered at 0
            Returns
            -------
            log_likelihood : float
            """

            num_components = data.get_shape().as_list()[1]
            log_likelihood = (
                -(log2pi * (num_components / 2.0))
                - tf.reduce_sum(tf.square(data) / 2.0, 1)
            )

            return log_likelihood

        log_p_given_z = bernoulli_log_joint(x_resampled)

        if importance_weighting:
            log_q_z = gaussian_likelihood(z, mean_encoder, log_variance_encoder)
            log_p_z = standard_gaussian_likelihood(z)

            regularization_term = log_p_z - log_q_z
        else:
            # Analytic solution to KL(q_z | p_z)
            p_z_q_z_kl_divergence =                 -self.beta*0.5 * tf.reduce_sum(1 
                                + log_variance_encoder
                                - tf.square(mean_encoder) 
                                - tf.exp(log_variance_encoder), 1) 

            regularization_term = -p_z_q_z_kl_divergence

        log_p_given_z_mc = tf.reshape(log_p_given_z, 
                                    [self.batch_size, monte_carlo_samples])
        regularization_term_mc = tf.reshape(regularization_term,
                            [self.batch_size, monte_carlo_samples])

        log_weights = log_p_given_z_mc + regularization_term_mc

        if importance_weighting:
            # Need to compute normalization constant for weights, which is
            # log (sum (exp(log_weights)))
            # weights_iw = tf.log(tf.sum(tf.exp(log_weights)))

            # Instead using log-sum-exp trick
            wmax = tf.reduce_max(log_weights, 1, keep_dims=True)

            # w_i = p_x/ q_z, log_wi = log_p_joint - log_qz
            # log ( 1/k * sum(exp(log w_i)))
            weights_iw = tf.log(tf.reduce_mean(tf.exp(log_weights - wmax), 1))
            objective = tf.reduce_mean(wmax) + tf.reduce_mean(weights_iw)
        else:
            objective = tf.reduce_mean(log_weights)

        return objective

    
    
    def partial_fit(self, X, epo):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        self.epoch= epo
        
        if self.warmup:
            #Warm-up Beta
            N_t = 30. #Number of epochs in warmup phase
            self.beta=self.epoch/N_t
        
        
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X, self.keep_prob: self.train_keep_prob})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X, self.keep_prob: 1.0})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu, self.keep_prob: 1.0})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X, self.keep_prob: 1.0})
    
def latent_covar(vae,n_samples=mnist.test.num_examples):
    test_data, _ =mnist.test.next_batch(n_samples)
    z_mean = vae.transform(test_data)
    return np.var(z_mean,0), np.mean(z_mean,0)

def count_significant(vae,threshold=1e-2,n_samples=mnist.test.num_examples):
    return np.greater(latent_covar(vae,n_samples),threshold).sum()


# In[5]:

def train(network_architecture, learning_rate=0.001,
            batch_size=100,
            training_epochs=10,
            display_step=1,
            train_keep_prob=1.0,
            batch_norm = True,
            warmup=False,
            iw=False):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size,
                                 train_keep_prob=train_keep_prob,
                                 batch_norm = batch_norm,
                                 warmup=warmup,
                                iw=iw)
    
    cost_list=np.ndarray(shape=(training_epochs,1),dtype=float)
    n_z=network_architecture['n_z']
    covar_list=np.ndarray(shape=(training_epochs,n_z),dtype=float)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs, epoch)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
            # Active Latent Dimension Count
            count = (count_significant(vae,n_samples=100))
            
        if np.isnan(avg_cost):
            print 'nan'
            nan = epoch
            break
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1),                   "cost=", "{:.9f}".format(avg_cost),                  "latent count=", "%d" % count
            sys.stdout.flush()
        cost_list.itemset((epoch,0),avg_cost)
        covari=latent_covar(vae,n_samples=100)
        for i in xrange(n_z):
            covar_list.itemset((epoch,i),covari[i])
        nan = epoch    
    return vae, cost_list, covar_list, nan


# In[27]:

def run_test(z_dim,keep_prob,b_normal,warmup, iw ,t_epochs=300, trial_num=0):
    network_architecture =         dict(n_hidden_recog_1=100, # 1st layer encoder neurons
             n_hidden_recog_2=100, # 2nd layer encoder neurons
             n_hidden_gener_1=100, # 1st layer decoder neurons
             n_hidden_gener_2=100, # 2nd layer decoder neurons
             n_input=784, # MNIST data input (img shape: 28*28)
             n_z=z_dim)  # dimensionality of latent space
    tf.reset_default_graph()
    vae, cost_list, covar_list, nan = train(network_architecture,learning_rate=0.001,
                                        batch_size=100,
                                        training_epochs=t_epochs,
                                        display_step=10,
                                        train_keep_prob=keep_prob,
                                        batch_norm = (b_normal==1),
                                        warmup = warmup,
                                        iw=iw)

    namestring='trials_final/{}.{}.{}.{}.{}.trial{}.endedat{}'.format(z_dim,keep_prob,b_normal,warmup,iw,trial_num,nan)
    pickle.dump({'cost':cost_list,'covar':covar_list},open(namestring+'.pkl','wb'))
    saver=tf.train.Saver()
    saver.save(vae.sess,namestring+'.ckpt')
    vae.sess.close()


# In[28]:

def load_and_run(z_dim,keep_prob,b_normal,warmup, iw ,t_epochs=300, trial_num=0,nan=299):
    network_architecture =         dict(n_hidden_recog_1=100, # 1st layer encoder neurons
             n_hidden_recog_2=100, # 2nd layer encoder neurons
             n_hidden_gener_1=100, # 1st layer decoder neurons
             n_hidden_gener_2=100, # 2nd layer decoder neurons
             n_input=784, # MNIST data input (img shape: 28*28)
             n_z=z_dim)  # dimensionality of latent space
    tf.reset_default_graph()
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=0.001, 
                                 batch_size=100,
                                 train_keep_prob=keep_prob,
                                 batch_norm = (b_normal==1),
                                 warmup=warmup,
                                iw=iw)
    namestring='trials_final/{}.{}.{}.{}.{}.trial{}.endedat{}'.format(z_dim,keep_prob,b_normal,warmup,iw,trial_num,nan)
#   vae.sess=tf.InteractiveSession()
    saver=tf.train.Saver()
    saver.restore(vae.sess,namestring+'.ckpt')
    return vae
    #covari=latent_covar(vae,n_samples=100)
    #vae.sess.close()


# In[29]:

#run_test(50,1,1,0,1,t_epochs=1)


# In[30]:

#load_and_run(50,1,1,0,1,nan=0)
