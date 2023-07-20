# Stellar Kinematics Neural Network
The repo contains the implementation, data and pretrained models for the kinematics project. Can be imported as a python package named `SKiNN`.

CUDA is required: make sure your setup is CUDA compatible

## Setup of SKiNN package
* Clone the repository into your environment: `git clone`
* Activate the envirionment you want the package to be in
* Go to your local lensing_odyssey_kinematics directory
* `pip install .` will download weights and install dependencies
* package should now be importable as `import SKiNN`

## How to test the network
* test the install with `python test_SKiNN.py`
* The notebook "SKiNN_package_example.ipynb" imports the package and generates a map
