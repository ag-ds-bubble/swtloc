import setuptools
import json
import sys
from urllib import request    
from pkg_resources import parse_version

""""
versioning : x[Major Update].x[Minor Update].x[Fixes]
"""

# Command to upload to testpypi : cls & rmdir /s /q build dist swtloc.egg-info & python setup_dev.py sdist & python setup_dev.py bdist_wheel & twine upload -r testpypi dist/*
# Installation :- pip install --extra-index-url https://test.pypi.org/simple/ swtloc


def _next_dev_version(pkg_name):
    try:
        url = f'https://test.pypi.org/pypi/{pkg_name}/json'
        releases = json.loads(request.urlopen(url).read())['releases']
        release = sorted(releases, key=parse_version)[-1]
        next_release = release.split('.')
        next_release[-1] = str(int(next_release[-1])+1)
        next_release = ".".join(next_release)
        return next_release
    except:
        return '1.1.100'

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
_next_version = _next_dev_version('swtloc')

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
                version=_next_version,
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




