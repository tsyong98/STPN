import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="STPN", # Replace with your own username
    version="0.0.2",
    author="Sin Yong Tan",
    author_email="tsyong98@gmail.com",
    description="Library for Spatiotemporal Pattern Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsyong98/STPN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)