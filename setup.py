from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Food web',
    url='https://github.com/jemff/food_web/',
    author='Emil Frølich',
    author_email='jaem@dtu.dk',
    # Needed to actually package something
    packages=['food_web_core'],
    # Needed for dependencies
    install_requires=['numpy', 'siconos', 'casadi'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='Apache',
    description='Food web basic functions',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)