from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))



VERSION = '0.0.2'
DESCRIPTION = 'na fe q vai funcionar'
LONG_DESCRIPTION = 'se chegou aqui ta faltando fe'

# Setting up
setup(
    name="HandoverRL",
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'matplotlib'],


)