from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()

setup(
    name="quickspacer",
    version="1.0.4",
    description="Korean spacing correction model that aims for fast speed and moderate accuracy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2.1"],
    url="https://github.com/cosmoquester/quickspacer.git",
    author="Park Sangjun",
    keywords=["spacer", "korean"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: Korean",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing",
    ],
    packages=find_packages(exclude=["tests"]),
    package_data={
        "quickspacer": [
            "resources/vocab.txt",
            "resources/default_saved_model/*/*",
            "resources/default_saved_model/*/*/*",
        ]
    },
)
