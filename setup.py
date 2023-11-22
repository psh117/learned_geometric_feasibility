## setup.py
from setuptools import find_packages, setup

setup(
    name='lgf',
    version='0.1',
    description='Learned Geometric Feasibility',
    packages=find_packages(include=['lgf', 'lgf.*']),
    install_requires=[
        'pytorch_lightning',
        'pyquaternion',
        'torch',
        'matplotlib',
        'scipy',
        'numpy'
    ]
)