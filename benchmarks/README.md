# Benchmark data

## Model Latency Table in ms
Tests are run on the file at `tests/assets/person.jpg` which contains a single person. The image size is 540x360 and the model picks the person up with no other predictions.

Note: 2080 Super runs out of memory past bs=20
| GPU | CPU | OS | Model | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=24 |
| --- | --- | -- | ----- | ---- | ---- | ---- | ---- | ----- | ----- |
| 2080 Super | R5 3600X | Ubuntu 22.04 (Native) | Mask-RCNN (ResNet50) | 45.5 | 85.1 | 164.5 | 322.8 | 651.1 | OOM |
| T4 | Intel Xeon (2 cores) | Ubuntu 22.04 (Colab) | Mask-RCNN (ResNet50) | 71.9 | 145.0 | 274.6 | 558.2 | 1145.8 | 1724.6 |

