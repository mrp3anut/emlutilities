from setuptools import setup, find_packages



setup(
    name="emlutilities",
    author="mrp3anut",
   
    
    url="https://github.com/mrp3anut/emlutilities",
    
    packages=find_packages(),
    setup_requires=[
    'numpy'],
    install_requires=[
    'numpy',
    'h5py',
    'pandas',
    'scipy',
    'scikit-learn'
    ], 
   )

