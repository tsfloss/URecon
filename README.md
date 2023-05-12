# URecon

URecon is a simple UNET architecture implemented in TensorFlow that can be used to reconstruct the initial conditions of N-body simulations from late time (e.g. z=0) density fields.
The weights trained on Quijote fiducial cosmology simulations, used for the results presented in 2305.07018 are provided.

Requirements:

 - numpy/scipy

- tensorflow

- tensorflow_addons (for instance normalization)

- tqdm

- pylians3 (only for measuring power spectrum of density fields)

