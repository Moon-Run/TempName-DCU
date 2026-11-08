from setuptools import find_packages, setup

setup(name="flux-dcu", version="0.1.0", packages=find_packages("python"),
      package_dir={"": "python"}, python_requires=">=3.10",
      data_files=[("share/flux-dcu/src/taco", ["src/taco/codec.hip"]),
                  ("share/flux-dcu", ["NOTICE", "LICENSE"])],
      description="Explicit DTK/HIP BLAS macroblock port of Flux and TACO designs")
