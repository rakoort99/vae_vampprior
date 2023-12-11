# VAE with a VampPrior
## Here's what I did:

1. added 2-layer VAE
2. added several 'new' priors. MBAP, KMeans cluster center, DBSCAN cluster exemplar (don't do this), and VampPrior *data*. 
   1. this required altering how data is loaded, creating new model attributes, and changing the training process to be individualized for each type of model. 
3. updated some code to run on modern pytorch. this wasn't necessary but was fun
4. added new model evaluation steps, such as histograms of posterior data in PCA space and active unit evaluation
5. learned a lot about pytorch. part of why I picked this project was so that I can learn to read and work with a real pytorch project, rather than sanitized, uncomplicated sample and tutorial code. 



## Below this point is work from the original paper
This is a PyTorch implementation of a new prior ("Variational Mixture of Posteriors" prior, or VampPrior for short) for the variational auto-encoder framework with one layer and two layers of stochastic hidden units as described in the following paper:
* Jakub M. Tomczak, Max Welling, VAE with a VampPrior, [arXiv preprint](https://arxiv.org/abs/1705.07120), 2017

## Requirements
The code is compatible with:
* `pytorch 0.2.0`

## Data
The experiments can be run on the following datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST);
* binary MNIST: the dataset is loaded from PyTorch;
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset could be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
* Histopathology Gray: the dataset could be downloaded from [link](https://github.com/jmtomczak/vae_householder_flow/tree/master/datasets/histopathologyGray);
* CIFAR 10: the dataset is loaded from PyTorch.

## Run the experiment
1. Set-up your experiment in `experiment.py`.
2. Run experiment:
```bash
python experiment.py
```
## Models
You can run a vanilla VAE, a one-layered VAE or a two-layered HVAE with the standard prior or the VampPrior by setting `model_name` argument to either: (i) `vae` or `hvae_2level` for MLP, (ii) `convvae_2level` for convnets, (iii) `pixelhvae_2level` for (ii) with a PixelCNN-based decoder, and specifying `prior` argument to either `standard` or `vampprior`.

## Citation

Please cite our paper if you use this code in your research:

```
@article{TW:2017,
  title={{VAE with a VampPrior}},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv},
  year={2017}
}
```

## Acknowledgments
The research conducted by Jakub M. Tomczak was funded by the European Commission within the Marie Skłodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).
