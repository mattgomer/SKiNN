
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    
setup(name='SKiNN',
  version='1.1',
  description='Stellar Kinematics Neural Network',
  author='Luca Biggio',
  url='https://github.com/lucabig/lensing_odyssey_kinematics',
  packages=find_packages(),
  include_package_data=True
      
 )
