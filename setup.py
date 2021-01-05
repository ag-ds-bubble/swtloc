import setuptools
import json
import sys
from urllib import request    
from pkg_resources import parse_version
from swtloc import __version__ as current_version

""""
versioning : x[Major Update].x[Minor Update].x[Fixes]
"""

# Command to upload to pypi : cls & rmdir /s /q build dist swtloc.egg-info & python setup.py sdist & python setup.py bdist_wheel & twine upload dist/*
# Installation :- pip install swtloc

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    try:
        lineiter = (line.strip() for line in open(filename))
        temp = [line.replace('==','>=') for line in lineiter if line and not line.startswith("#")]
        return [k for k in temp if 'scikit-learn' not in k]
    except:
        return []

# Constants
REQS = parse_requirements('requirements.txt')

with open("README.md", "r") as fh:
    long_description = fh.read()


CLASSIFIERS = ["Programming Language :: Python :: 3",
                "Development Status :: 2 - Pre-Alpha",
                "Intended Audience :: Developers",
                "Intended Audience :: Education",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: BSD License",
                "Operating System :: OS Independent",
                "Topic :: Software Development :: Libraries"]

# Package Setup
setuptools.setup(name="swtloc",
                version=current_version,
                author="Achintya Gupta",
                author_email="ag.ds.bubble@gmail.com",
                description="Python Library for Stroke Width Transform",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/ag-ds-bubble/swtloc",
                packages=setuptools.find_packages(),
                include_package_data = True,
                install_requires=REQS,
                classifiers=CLASSIFIERS,
                python_requires='>=3.6')

