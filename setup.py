import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-dft",
    version="0.1",
    author="Mihail Bogojeski",
    author_email="m.bogojeski@tu-berlin.de",
    description="A package for density functional approximation using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MihailBogojeski/ml-dft",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
