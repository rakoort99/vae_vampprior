import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
#=======================================================================================================================
def plot_histogram( x, dir, mode ):

    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, density=True, facecolor='blue', alpha=0.5)

    plt.xlabel('Log-likelihood value')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.savefig(dir + 'histogram_' + mode + '.png', bbox_inches='tight')
    plt.close(fig)

def qz_pca_scatter(model, test_data, labels, dir, mode):
    if '2' in model.args.model_name:
        _, _, z1_q, _, _, z2_q, _, _, _, _, _ = model.forward(test_data)
        z1 = z1_q.cpu().detach().numpy()
        z2 = z2_q.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()


        # z1
        fig = plt.figure()
        # get pca of latents
        V = np.linalg.svd(z1)[2][:2,:].T
        proj = z1.dot(V)
        for l in np.unique(labels):
            i = np.where(labels == l)
            hmm = plt.scatter(proj[:,0][i], proj[:,1][i], label=l, alpha=0.5)
        fig.legend()
        plt.savefig(dir + 'qz1_scatter_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        hmm = plt.hist2d(proj[:,0], proj[:,1], bins=[25,25], density=True)

        plt.savefig(dir + 'qz1_hist_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)

        # z2
        fig = plt.figure()
        # get pca of latents
        V = np.linalg.svd(z2)[2][:2,:].T
        proj = z2.dot(V)
        for l in np.unique(labels):
            i = np.where(labels == l)
            hmm = plt.scatter(proj[:,0][i], proj[:,1][i], label=l, alpha=0.5)
        fig.legend()
        plt.savefig(dir + 'qz2_scatter_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        hmm = plt.hist2d(proj[:,0], proj[:,1], bins=[25,25], density=True)

        plt.savefig(dir + 'qz2_hist_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)


    else:
        z = model.q_z_layers(test_data)
        z = z.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        fig = plt.figure()

        # get pca of latents
        V = np.linalg.svd(z)[2][:2,:].T
        proj = z.dot(V)
        for l in np.unique(labels):
            i = np.where(labels == l)
            hmm = plt.scatter(proj[:,0][i], proj[:,1][i], label=l, alpha=0.5)
        fig.legend()
        plt.savefig(dir + 'qz_scatter_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        hmm = plt.hist2d(proj[:,0], proj[:,1], bins=[25,25], density=True)

        plt.savefig(dir + 'qz_hist_' + mode + '.png', bbox_inches='tight')
        plt.close(fig)

#=======================================================================================================================
def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):

    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample[0:25]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)