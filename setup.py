import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FSS-tools",
    version="0.1.0",
    install_requires=["torch>=1.10", "gpytorch>=1.6",],
    author="Kenji Harada",
    author_email="harada.kenji.8e@kyoto-u.ac.jp",
    description="Finite-size scaling tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KenjiHarada/FSS-tools",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
