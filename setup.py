"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="filib",
    use_scm_version={"write_to": "src/filib/_version.py"},
    description="Factor Investing Library",
    long_description=long_description,
    url="https://github.com/prmkowalski/filib",
    author="Paweł Kowalski",
    author_email="prm.kowalski@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["pandas<1.4"],
    extras_require={"test": ["coverage[toml]>=5.0.2", "matplotlib", "pytest"]},
    setup_requires=["setuptools_scm"],
    project_urls={
        "Bug Tracker": "https://github.com/prmkowalski/filib/issues",
        "Source Code": "https://github.com/prmkowalski/filib",
    },
)
