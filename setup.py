
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
setup(name='SKiNN',
  version='1.0',
  description='Stellar Kinematics Neural Network',
  author='Luca Biggio',
  url='https://github.com/lucabig/lensing_odyssey_kinematics',
  packages=['SKiNN'],
 )
