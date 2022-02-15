""" setup script """
import setuptools

exec(open("momlevel/version.py").read())

setuptools.setup(version=__version__)
