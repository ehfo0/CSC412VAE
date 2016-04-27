import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import vae_save
mnist=vae_save.mnist
font=18
def lattice(mean_dims,indicies,d_i,full_dim):
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
    im=plt.imshow(canvas, origin="upper")
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('plots/img/vis_latent%s_%s.png' % (full_dim, d_i), dpi=300)

x_sample = mnist.test.next_batch(100)[0]
for dim in [2,50]:
    vae=vae_save.load_and_run(dim,1.0,0,0,1,trial_num=0,nan=299)
    x_reconstruct = vae.reconstruct(x_sample)
    if dim==2:
        plt.figure(figsize=(12, 12))
    for i in range(5):
        if dim==2:
            sp=plt.subplot(5, 3, 3*i + 1)
            sp.axes.get_xaxis().set_visible(False)
            sp.axes.get_yaxis().set_visible(False)
            plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
            if i==0:
                plt.title("Test input",fontsize=font)
            sp=plt.subplot(5, 3, 3*i + 2)
            sp.axes.get_xaxis().set_visible(False)
            sp.axes.get_yaxis().set_visible(False)
            plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
            if i==0:
                plt.title("2D Reconstruction",fontsize=font)
        elif dim==50:
            sp=plt.subplot(5, 3, 3*i + 3)
            sp.axes.get_xaxis().set_visible(False)
            sp.axes.get_yaxis().set_visible(False)
            plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
            if i==0:
                plt.title("50D Reconstruction",fontsize=font)
    vae.sess.close()
plt.subplots_adjust(wspace=0,hspace=0)
plt.savefig('plots/img/vis_recon.png',dpi=300)
for dim in [2,50]:
    vae=vae_save.load_and_run(dim,1.0,0,0,1,trial_num=0,nan=299)
    covar_dims, mean_dims=vae_save.latent_covar(vae)
    indices=np.argsort(covar_dims)
    x_sample2, y_sample2 = mnist.test.next_batch(5000)
    z_mu = vae.transform(x_sample2)
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    ticks = np.linspace(0.5,9.5,10)
    bounds = np.linspace(0,10,11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # make the scatter
    scat = ax.scatter(z_mu[:, 1], z_mu[:, 0],c=np.argmax(y_sample2, 1),cmap=cmap, norm=norm)
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i')
    plt.savefig('plots/img/vis_scatter%s.png' % dim,dpi=300)
    if dim==2:
        lattice(mean_dims,[0,1],1,2)
    elif dim==50:
        for d_i in [1,49]:
            lattice(mean_dims,indices,d_i,50)
    vae.sess.close()
