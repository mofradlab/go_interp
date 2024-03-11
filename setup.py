import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="go_ml", 
    version="0.0.1",
    author="Andrew Dickson",
    author_email="amdickson@berkeley.edu",
    description="Package for GO models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amdson/gointerp",
    packages=["go_ml"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires='>=3.9',
)