from setuptools import setup, find_packages

setup(
    name="dCFE",  # Name of your package
    version="0.1",  # Version number
    description="CFE in torch",  # Short description
    url="https://github.com/NWC-CUAHSI-Summer-Institute/dCFE",  # URL to the github repo
    author="Your Name",  # Your name
    author_email="raraki8159@sdsu.edu",  # Your email
    license="MIT",  # License type
    packages=find_packages(),  # Automatically find all packages
    install_requires=[],  # List of dependencies (as strings)
    zip_safe=False,  # Not necessary, set to False if unsure
)
