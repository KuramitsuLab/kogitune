from setuptools import setup, find_packages

"""
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
"""


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="kogitune",
    version="0.3.1.2024.3.26",
    license="Apache",
    author="Kimio Kuramitsu",
    description="The KOGItune 🦊 LLM Project",
    url="https://github.com/kuramitsulab/kogitune",
    packages=["kogitune", 
              'kogitune.datasets',
              'kogitune.filters', 
              'kogitune.stores',
              'kogitune.trainers',  
              'kogitune.metrics'],
    package_dir={"kogitune": "kogitune"},
    package_data={"kogitune": ["*/*"]},
    install_requires=_requires_from_file("requirements.txt"),
    entry_points={
        "console_scripts": [
            "kogitune=kogitune.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Education",
    ],
)
