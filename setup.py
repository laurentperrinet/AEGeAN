#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    import AEGEAN
    version = AEGEAN.__version__
except:
    version = '20191014'

setup(
    name='AEGEAN',
    version=version,
    packages=find_packages(),
    author="",
    author_email="",
    description="",
    long_description=open('README.md').read(),
    include_package_data=True,
    url='',
    classifiers=[
        "Programming Language :: Python",
    ],
)
