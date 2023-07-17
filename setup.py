
import os
import urllib.request

try:
    from setuptools import setup, find_packages, find_namespace_packages
except ImportError:
    from distutils.core import setup

weights_url = "https://dox.uliege.be/index.php/s/mtphEyrQ5h4IRF8/download"
weights_file = "SKiNN/weights/upsampling_cosmo_gen_norm_1channel_new.ckpt"
weights_dir = os.path.dirname(weights_file)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

if not os.path.exists(weights_file):
    print('Downloading weights...')
    urllib.request.urlretrieve(weights_url, weights_file)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

    
setup(name='SKiNN',
  version='1.1',
  description='Stellar Kinematics Neural Network',
  author='Luca Biggio',
  url='https://github.com/lucabig/lensing_odyssey_kinematics',
  packages=find_packages()+find_namespace_packages(),
  include_package_data=True,
  install_requires=requirements
      
 )
