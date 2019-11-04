#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import AEGEAN

setup(
    name='AEGEAN',
    version=AEGEAN.__version__,
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
