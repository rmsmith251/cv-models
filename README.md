# cv-models

This repo is a collection of various models that I've implemented in PyTorch. There are many great implementations of these models already, this is just a practice repo for me but feel free to contribute.

## Development
To get started, clone the repo with
```bash
git clone https://github.com/rmsmith251/cv-models.git
```
If you want to contribute, please install the dev requirements and install the pre-commit hooks using:
```bash
pip install -e .[dev]
pre-commit install
```

Or you can install directly from GitHub using
```bash
pip install 'models'@git+https://github.com/rmsmith251/cv-models

# To install with developer packages
pip install 'models[dev]'@git+https://github.com/rmsmith251/cv-models
```

## Benchmarks
I will be keeping a table of benchmark numbers for the various models along with the relevant hardware specs. This table can be found [here](benchmarks/README.md).

If you would like to contribute your own benchmarks, please use the following guidelines:
1. Ensure anything using the GPU (e.g. YouTube, games, video call, etc) is shut down. Web browsers are fine to be open as long as there isn't a graphics intensive task in the background.
2. Run the benchmark using `pytest benchmarks/models/...`.
3. Update the table in a PR and include a screenshot of the command line table that is generated. Please be sure to include your hardware and OS so that others can easily compare.