# lensing odyssey kinematics project
The repo contains the implementation, data and pretrained models for the kinematics project. Can be imported as a python package named `SKiNN`.

## Setup of SKiNN package
* Clone the repository into your environment: `git clone`
* Download the correct weights from ____, called 'upsampling_cosmo_gen_norm_1channel_new-epoch=1109-valid_loss=0.00.ckpt'. Put this file in the 'SKiNN/weights/' directory.
* Activate the envirionment you want the package to be in
* install dependencies from `requirements.txt`
* CUDA is required: make sure your version of torch is cuda compatible
* Run `python setup.py install` to install as SKiNN package
* package should now be importable as `import SKiNN`

## How to test the network
* The notebook `SKiNN_package_example` imports the package and generates a map
* The notebook `interface.ipynb` shows how to load and test the package without importing SKiNN

  
## Setup without importing SKiNN package (old)
* Clone the repository: `git clone`
* Create a virtual environment: `python -m virtualenv cosmo` # Note, we used python 3.8.5 for this project
* Activate the virtual environment: `source cosmo/bin/activate`
* Install dependencies: `pip install -r requirements.txt`
* Additionally install `pytorch` from the official website
