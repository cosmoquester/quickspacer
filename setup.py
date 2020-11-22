from setuptools import find_packages, setup

setup(
    name="quickspacer",
    version="1.0.2",
    description="Korean spacing correction model that aims for fast speed and moderate accuracy.",
    python_requires=">=3.7",
    install_requires=["tensorflow>=2.3.0"],
    url="https://github.com/psj8252/quickspacer.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
    package_data={
        "quickspacer": [
            "resources/vocab.txt",
            "resources/default_saved_model/*/*",
            "resources/default_saved_model/*/*/*",
        ]
    },
)
