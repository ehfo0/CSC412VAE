import numpy as np
import matplotlib.pyplot as plt
import vae_save
mnist=vae_save.mnist
vae=vae_save.load_and_run(50,1.0,0,0,1,trial_num=0,nan=299)
covar_dims, mean_dims=vae_save.latent_covar(vae)
indices=np.argsort(covar_dims)
print indices
# In[131]:

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))
for i in range(5):

    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()


# In[13]:


x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()
plt.savefig('vis1.png',dpi=300)


# In[66]:
for d_i in [1,49]:
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    z_dims=mean_dims
    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_dims[indices[0]]=xi
            z_dims[indices[d_i]]=yi
            z_mu = np.array([z_dims]*100)
            x_mean = vae.generate(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    #plt.show()
    plt.tight_layout()
    plt.savefig('vis2.%s.png' % d_i, dpi=300)
