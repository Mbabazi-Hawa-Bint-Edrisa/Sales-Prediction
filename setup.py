from setuptools import find_packages, setup
from typing import List
import os

HYPHEN_DOT_E = "-e ."
def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n", "") for req in requirements]
            if HYPHEN_DOT_E in requirements:
                requirements.remove(HYPHEN_DOT_E)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using empty requirements list.")
    return requirements

setup(
    name='Sales Prediction Project',
    version='0.1.0',
    author='Mbabazi Hawa Bint Edrisa',
    author_email='mbabaziedrisa0@gmail.com',
    description='A project to predict sales using machine learning models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt')),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)