# Stellar Kinematics Neural Network
The repo contains the implementation, data and pretrained models for the kinematics project. Can be imported as a python package named `SKiNN`.

CUDA is required: make sure your setup is CUDA compatible

## 1/15/26 updates
- URL changed for weights, now available on ULiege Dataverse: Gomer, Matthew R.; Ertl, Sebastian; Biggio, Luca; Wang, Hang; Galan, Aymeric; Van de Vyvere, Lyne; Sluse, Dominique; Vernardos, Georgios; Suyu, Sherry H., 2026, "Weights and training data for SKiNN (Stellar Kinematics Neural Network)", https://doi.org/10.58119/ULG/WZFXYD, ULiège Open Data Repository, V1
- Made minimal changes necessary to make download work when test_SKiNN.py is called, but just in case that fails, weights are accessible through the above link.

## Setup of SKiNN package
* Clone the repository into your environment: `git clone`
* Activate the envirionment you want the package to be in
* Go to your local SKiNN directory
* `pip install -r SKiNN/requirements.txt` will install dependencies
* package should now be importable as `import SKiNN`
* weights will be downloaded when first called. Run the test script below.
* confirmed to work in a Colab environment on 1/15/26, not guaranteed to be maintained since then.

## How to test the network
* test the install with `python test_SKiNN.py`
* The notebook "SKiNN_package_example.ipynb" imports the package and generates a map
