import re
from setuptools import setup
import subprocess


def git_version_tag():
    """
    Retrieve the latest version tag and corresponding hash from git repo.
    :return: Hash of the tag and version tag.
    """

    try:
        refs = subprocess.check_output(["git", "show-ref", "--abbrev", "--tags"]) \
            .decode("utf-8") \
            .strip() \
            .splitlines()
    except subprocess.CalledProcessError:
        # Either not a git repo or there is no tag.
        return 0, "0.0"

    pattern = re.compile(r"^v(\d+\.\d+(\.\d+)?)$")
    for tag_hash, tag in map(lambda pair: (pair[0], pair[1].rsplit("/", 1)[1]),
                             map(lambda line: line.split(" "),
                                 refs[::-1])):
        match = pattern.match(tag)
        if match:
            return tag_hash, match[1]

    # No tag matches the pattern.
    return 0, "0.0"


def get_version():
    """
    Build version string based on git repository.
    :return: Version string.
    """

    tag_hash, version = git_version_tag()
    current_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]) \
        .decode("utf-8") \
        .strip()
    if tag_hash != current_hash:
        version = version + ".dev" + current_hash

    return version


setup(
    name="pentinsula",
    version=get_version(),
    packages=["pentinsula"],
    install_requires=["h5py",
                      "numpy"],

    author="Jan-Lukas Wynen",
    author_email="j-l.wynen@hotmail.de",
    description="h5py utilities for time series data in HDF5",
    keywords="hdf5 time_series h5py",
    url="https://github.com/jl-wynen/pentinsula",
    project_urls={
        "Source Code": "https://github.com/jl-wynen/pentinsula",
    },
    # TODO fill in
    # python_requires=">=",
    # classifiers=[
    #
    # ]
)
