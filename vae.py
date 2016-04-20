#Adapted from code by Jan Hendrik Metzen: https://jmetzen.github.io/2015-11-27/vae.html
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pylab
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
#matplotlib inline
np.random.seed(0)
tf.set_random_seed(0)
# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
class VariationalAutoencoder(object):
    """Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details."""

    def __init__(self, n_layers, layer_n, latent_dim, input_dim, transfer_fct=tf.nn.softplus, \
                 learning_rate=0.001, batch_size=100,restore=False):
        self.n_layers = n_layers
        self.layer_nodes = layer_n
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_epochs = 0
        self.sess = tf.InteractiveSession()
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.dropout=tf.placeholder(tf.int32, [n_layers])
        # Create autoencoder network
        self._create_network(n_layers, layer_n, latent_dim, input_dim)
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        self.saver=tf.train.Saver()
        if not restore:
            # Initializing the tensor flow variables
            init = tf.initialize_all_variables()

            # Launch the session
            self.sess.run(init)
    
    
    def _create_network(self,n_layers, layer_n, latent_dim, n_input):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(n_layers, layer_n, n_layers, layer_n, n_input, latent_dim)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(len(network_weights["weights_recog"]), \
                                      network_weights["weights_recog"], \
                                      network_weights["biases_recog"], \
                                      network_weights["w_recog_out_mean"], \
                                      network_weights["w_recog_out_log_sigma"], \
                                      network_weights["b_recog_out_mean"], \
                                      network_weights["b_recog_out_log_sigma"])

        # Draw one sample z from Gaussian distribution
        n_z = self.latent_dim
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(len(network_weights["weights_gener"]), \
                                    network_weights["weights_gener"], \
                                    network_weights["biases_gener"], \
                                    network_weights["w_gener_out_mean"], \
                                    network_weights["w_gener_out_log_sigma"], \
                                    network_weights["b_gener_out_mean"], \
                                    network_weights["b_gener_out_log_sigma"])
            
    def _initialize_weights(self, n_hidden_recog_layers, n_hidden_recog_dim, 
                            n_hidden_gener_layers,  n_hidden_gener_dim, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog']=[]
        all_weights['weights_recog'].append(tf.Variable(xavier_init(n_input, n_hidden_recog_dim)))
        for i in xrange(1,n_hidden_recog_layers):
            all_weights['weights_recog'].append(tf.Variable(xavier_init(n_hidden_recog_dim, n_hidden_recog_dim)))
        all_weights['w_recog_out_mean']= tf.Variable(xavier_init(n_hidden_recog_dim, n_z))
        all_weights['w_recog_out_log_sigma']= tf.Variable(xavier_init(n_hidden_recog_dim, n_z))
        all_weights['biases_recog']=[]
        for i in xrange(0,n_hidden_recog_layers):
            all_weights['biases_recog'].append(tf.Variable(tf.zeros([n_hidden_recog_dim], dtype=tf.float32)))
        all_weights['b_recog_out_mean']=(tf.Variable(tf.zeros([n_z], dtype=tf.float32)))
        all_weights['b_recog_out_log_sigma']=tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        all_weights['weights_gener']=[]
        all_weights['weights_gener'].append(tf.Variable(xavier_init(n_z, n_hidden_gener_dim)))
        for i in xrange(1,n_hidden_gener_layers):
            all_weights['weights_gener'].append(tf.Variable(xavier_init(n_hidden_gener_dim, n_hidden_gener_dim)))
        all_weights['w_gener_out_mean']= tf.Variable(xavier_init(n_hidden_gener_dim, n_input))
        all_weights['w_gener_out_log_sigma']= tf.Variable(xavier_init(n_hidden_gener_dim, n_input))
        all_weights['biases_gener'] = []
        for i in xrange(0,n_hidden_gener_layers):
            all_weights['biases_gener'].append(tf.Variable(tf.zeros([n_hidden_gener_dim], dtype=tf.float32)))
        all_weights['b_gener_out_mean']=tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        all_weights['b_gener_out_log_sigma']=tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        
        return all_weights
            
    def _recognition_network(self, n_layers, weights, biases, \
                             w_recog_out_mean, w_recog_out_log_sigma, b_recog_out_mean, b_recog_out_log_sigma):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        curr_layer=self.x
        for i in xrange(n_layers):
            curr_layer = self.transfer_fct(tf.add(tf.matmul(curr_layer, weights[i]), biases[i]))
        z_mean = tf.add(tf.matmul(curr_layer, w_recog_out_mean), \
                        b_recog_out_mean)
        z_log_sigma_sq = \
            tf.add(tf.matmul(curr_layer, w_recog_out_log_sigma), \
                   b_recog_out_log_sigma)
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, n_layers, weights, biases, \
                           w_gener_out_mean, w_gener_out_log_sigma, b_gener_out_mean, b_gener_out_log_sigma):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        curr_layer=self.z
        for i in range(n_layers):
            curr_layer = self.transfer_fct(tf.add(tf.matmul(curr_layer, weights[i]), \
                                                      biases[i]))
        z_mean = tf.add(tf.matmul(curr_layer, w_gener_out_mean), \
                        b_gener_out_mean)
        z_log_sigma_sq = \
            tf.add(tf.matmul(curr_layer, w_gener_out_log_sigma), \
                   b_gener_out_log_sigma)
                                      
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(curr_layer, w_gener_out_mean), \
                                 b_gener_out_mean))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        #self.cost = self._evidence_lower_bound();
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X,self.dropout: [0]*n_layers})#tf.zeros([1,self.n_layers])})
        return cost
    
    def transform(self, X,layer_dropout=[]):
        """Transforpm data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None,layer_dropout=[]):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = [np.random.normal(size=self.latent_dim)]
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        feed_dict={self.z: z_mu}
        for l in layer_dropout:
            print self.network_weights["weights_recog"][l].shape()
            self.weights[l]=tf.Variable(tf.zeros(self.layer_nodes,self.layer_nodes))
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict=feed_dict)
    
    def reconstruct(self, X,layer_dropout=[]):
        """ Use VAE to reconstruct given data. """
        self.dropout=layer_dropout
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})
    
    def _evidence_lower_bound(self,
                              monte_carlo_samples=1,
                              importance_weighting=False,
                              tol=1e-5):
        """
            Variational objective function

            ELBO = E(log joint log-likelihood) - E(log q)
                 = MC estimate of log joint - Entropy(q)

        """
        x_resampled = tf.tile(self.x, tf.constant([monte_carlo_samples, 1]))

        # Forward pass of data into latent space
        mean_encoder, log_variance_encoder = self.z_mean, self.z_log_sigma_sq#self._encode(x_resampled)

        #random_noise = tf.random_normal(
        #    (self.batch_size * monte_carlo_samples, self.latent_dim), 0, 1, dtype=tf.float32)

        # Reparameterization trick of re-scaling/transforming random error
        std_dev = tf.sqrt(tf.exp(log_variance_encoder))
        z = mean_encoder + std_dev# * random_noise

        # Reconstruction/decoding of latent space
        mean_decoder = self.x_reconstr_mean#self._generate(z)

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
            p_z_q_z_kl_divergence = \
                -0.5 * tf.reduce_sum(1 
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
    
def train(n_layers, layer_n, latent_dim, input_dim, learning_rate=0.001, \
          batch_size=100, training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(n_layers, layer_n, latent_dim, input_dim, \
                             learning_rate=learning_rate, \
                             batch_size=batch_size)
    # Training cycle
    vae.training_epochs=training_epochs
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
               "cost=", "{:.9f}".format(avg_cost)
    return vae

def visualize_2d_gener(vae_2d,test_data=mnist.test,layer_dropout=[]):
    if vae_2d.latent_dim!=2:
        print "Need a two-dimensional VAE to visualize"
        return
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean = vae_2d.generate(z_mu,layer_dropout=layer_dropout)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    pylab.show()

def visualize_2d_recon(vae_2d,test_data=mnist.test,layer_dropout=[]):
    if vae_2d.latent_dim!=2:
        print "Need a two-dimensional VAE to visualize"
        return
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    x_sample = test_data.next_batch(100)[0]
    x_reconstruct = vae_2d.reconstruct(x_sample,layer_dropout=layer_dropout)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        x_sample, y_sample = test_data.next_batch(5000)
        z_mu = vae_2d.transform(x_sample,layer_dropout=layer_dropout)
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    pylab.show()

def filename(n_layers,layer_nodes,latent_dim,input_dim,training_epochs,learning_rate=0.001,batch_size=100):
    return "model{:d}.{:d}.{:d}.{:d}.{:d}.ckpt".format(n_layers,layer_nodes,latent_dim,input_dim,training_epochs)

    
def save_model(vae):
    vae.saver.save(vae.sess,filename(vae.n_layers,vae.layer_nodes,vae.latent_dim,vae.input_dim,vae.training_epochs))
    
def load_model(n_layers,layer_nodes,latent_dim,input_dim,training_epochs,learning_rate=0.001,batch_size=100):
    vae = VariationalAutoencoder(n_layers, layer_nodes, latent_dim, input_dim,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 restore=True)
    vae.saver=tf.train.Saver()
    vae.sess=tf.Session()
    vae.saver.restore(vae.sess,filename(n_layers,layer_nodes,latent_dim,input_dim,training_epochs,learning_rate=0.001,batch_size=100))
    return vae

def latent_covar(vae,n_samples=mnist.test.num_examples):
    test_data, _ =mnist.test.next_batch(n_samples)
    z_mean = vae.transform(test_data)
    return np.var(z_mean,0)

def count_significant(vae,threshold=1e-2,n_samples=mnist.test.num_examples):
    return np.greater(latent_covar(vae,n_samples),threshold).sum()

if __name__=='__main__':
    #Number of layers
    n_layers=2
    #Nodes per layer
    layer_n=10
    latent_dim=10
    input_dim=784
    training_epochs=20
    #vae = train(n_layers,layer_n,latent_dim,input_dim,training_epochs=training_epochs)
    #save_model(vae)
    vae=load_model(n_layers,layer_n,latent_dim,input_dim,training_epochs)
    print count_significant(vae)
    
    latent_dim=20
    #vae = train(n_layers,layer_n,latent_dim,input_dim,training_epochs=training_epochs)
    #save_model(vae)
    vae=load_model(n_layers,layer_n,latent_dim,input_dim,training_epochs)
    print count_significant(vae)
