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
    version="0.1.10.21",
    license="Apache",
    author="Kimio Kuramitsu",
    description="The Kogitune 🦊 LLM Project",
    url="https://github.com/kuramitsulab/kogitune",
    packages=["kogitune"],
    package_dir={"kogitune": "kogitune"},
    package_data={"kogitune": ["*/*"]},
    install_requires=_requires_from_file("requirements.txt"),
    entry_points={
        "console_scripts": [
            "kogitune_store=kogitune.cli:main_store",
	        "kogitune_update=kogitune.cli:main_update",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Education",
    ],
)
