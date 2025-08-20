# Convolutional Self Attention
Adapted CSA from this NVIDIA Blog post: [here](https://developer.nvidia.com/blog/emulating-the-attention-mechanism-in-transformer-models-with-a-fully-convolutional-network/)

# Benchmarking
All models were run on T4 gpu instances in google colab.

The tests are done on the built-in torch MNIST dataset; with input sizes of 28x28. The tests are sorted into two categories:

- Classification of numbers
- Generation of images with classification as a secondary objective.

Included in each test is a video of the evolution of the Principle Component Analysis of the Latent Space of each model.
#TODO: Display comparison of generated numbers, as well as comparisons of per-class AUROC

# Results

| Model Type      | MNIST NLL    | MNIST Accuracy | # of Parameters | Runtime (s) |
| --------------- | ------------ | -------------- | --------------- | ----------- |
|CSA 1 Layer      | 1.405e-1     |                |                 |             |
|CSA 2 Layers     | 2.768e-1     |                |                 |             |
|LSA 1 Layer      | 2.147e-1     |                |                 |             |
|LSA 2 Layers     | 6.309e-1     |                |                 |             |
| Simple CNN      | **7.468e-7** |                |                 |             |
|torch example CNN| 2.18e-5      |                |                 |             |
