from setuptools import setup
import os

def read_requirements():
    """Reads the requirements from requirements.txt, handling comments."""
    reqs_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    with open(reqs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # Strip leading/trailing whitespace from the whole line

            # Skip empty lines and lines that are entirely comments
            if not line or line.startswith('#'):
                continue

            # Remove inline comments
            # Find the first '#' character
            comment_start = line.find('#')
            if comment_start != -1:
                # Take the part of the line before the comment, and strip trailing whitespace
                line = line[:comment_start].rstrip()

            # Add the cleaned line if it's not empty after removing the comment
            if line:
                requirements.append(line)

    return requirements

# Minimal setup to be referenced from pyproject.toml to use requirements.txt for dependencies
# All metadata (name, version, description, etc.) is read from pyproject.toml.
# Only install_requires is defined here because it's declared as dynamic
# in pyproject.toml and setuptools falls back to looking for it here.
setup(
    install_requires=read_requirements(),
)