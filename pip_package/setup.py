"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


setup(
    name="tf-locality-aware-nms",
    version="0.0.1",
    description="Locality-Aware NMS as a Tensorflow op.",
    long_description="""
    An implementation of Locality-Aware NMS as described in 
    EAST: An Efficient and Accurate Scene Text detector (https://arxiv.org/abs/1704.03155)
    as a Tensorflow op meaning it can be used as any other Tensorflow function. It can also 
    be included in a Tensorflow Serving build to make it available for serving.
    """,
    author="John Pertoft",
    author_email="john.pertoft@gmail.com",
    url="https://github.com/johnPertoft/locality-aware-nms",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
)
