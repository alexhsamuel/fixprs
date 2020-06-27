import glob
from   numpy.distutils.misc_util import get_numpy_include_dirs
from   pathlib import Path
import setuptools

with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fixprs",
    version="0.1.0",
    description="Fast parsers for textual data formats",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/alexhsamuel/fixprs",
    author="Alex Samuel",
    author_email="alex@alexsamuel.net",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="",
    project_urls={
    },

    package_dir={"": "src"},  # Optional
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6, <4",
    install_requires=[],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["pytest"],
    },
    package_data={
    },
    data_files=[
    ],
    entry_points={  # Optional
        # "console_scripts": [
        #     "sample=sample:main",
        # ],
    },

    ext_modules     =[
        setuptools.Extension(
            "fixprs.ext",
            extra_compile_args  =["-std=c++14"],
            include_dirs        =[
                "./vendor/fast_double_parser/include",
                "./vendor/ThreadPool",
                *get_numpy_include_dirs(),
            ],
            sources             =glob.glob("src/fixprs/ext/*.cc"),
            library_dirs        =[],
            libraries           =[],
            depends             =glob.glob("src/fixprs/ext/*.hh"),
        ),
    ],

)
