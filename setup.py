from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ARTST',
    version='0.1',
    packages=find_packages(),
     package_data={
        '': ['*.*'], 
    },
    install_requires=required,  
    long_description_content_type='text/markdown',
    url='https://github.com/1eurs/ARTST',
)
