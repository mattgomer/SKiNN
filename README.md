# lensing odyssey kinematics project
The repo contains the implementation, data and pretrained models for the kinematics project. Can be imported as a python package named `SKiNN`.

CUDA is required: make sure your setup is CUDA compatible

## Setup of SKiNN package
* Clone the repository into your environment: `git clone`
* Activate the envirionment you want the package to be in
* Go to your local lensing_odyssey_kinematics directory
* "pip install ." will download weights and install dependencies
* package should now be importable as `import SKiNN`
* test the install with "python test_SKiNN.py"

## How to test the network
* The notebook `SKiNN_package_example` imports the package and generates a map
* The notebook `interface.ipynb` shows how to load and test the package without importing SKiNN

