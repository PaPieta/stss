from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="st2ss",
      version="0.1.0",
      author="Pawel Pieta",
      author_email="papi@dtu.dk",
      description="2D and 3D structure tensor scale space implementation for Python.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/PaPieta/st2ss",
      packages=find_packages(),
    # #   [
    #       "structure_tensor",
    #   ],
      python_requires='>=3',
      install_requires=["numpy>=1.16", "scipy>=1.3"],
    #   extras_require={"CuPy": ["cupy>=8"]},
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
      ])
