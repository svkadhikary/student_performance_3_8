from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    
    HYPHEN_E_DOT = "-e ."
    requirements = []

    with open(file_path) as req_file:
        requirements = req_file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(

    name = "Student Performance Prediction",
    version = "0.0.1",
    author = "Souvik Adhikary",
    author_email = "svkadhikary7@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)
