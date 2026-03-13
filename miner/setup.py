import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="score",
    version='0.1.0',
    python_requires=">=3.8",
    description="",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tmoklc/miner",
    author="tmoklc",
    author_email="tmoklc@gmail.com",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "supervision",
        "numpy",
        "opencv-python",
        "transformers",
        "umap-learn==0.5.7",
        "scikit-learn",
        "tqdm",
        "sentencepiece",
        "protobuf"
    ],
    extras_require={
        'tests': [
            'pytest',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)