from setuptools import find_packages, setup

setup(
    name="hgsprediction",
    version="0.1.0",
    description="Hand-grip strength prediction in UK Biobank",
    url="https://github.com/KNazarzadeh/hgsprediction.git",
    author="Kimia Nazarzadeh",
    packages=find_packages(),
    install_requires=[
        "seaborn",
        "nilearn>=0.9.0",
        "datalad>=0.15.0"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={"": ["data/*"]},
)
