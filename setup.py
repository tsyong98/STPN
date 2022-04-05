import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="STPN", # Replace with your own username
    version="0.0.5",
    author="Sin Yong Tan",
    author_email="tsyong98@gmail.com",
    description="Library for Spatiotemporal Pattern Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsyong98/STPN",
    project_urls={
        "Bug Tracker": "https://github.com/tsyong98/STPN/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires='>=3.6',
    install_requires=[
        'numpy', 
        'pandas',
        'matplotlib',
        'sklearn'
    ]
)