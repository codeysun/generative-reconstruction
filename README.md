# Reimagine the World with 3D Generative Reconstruction

[Slides](https://docs.google.com/presentation/d/1Q3ksUNysvB-zoIKSnR7MmscWLzR4cuI4X7SXcms_wRc/edit?usp=sharing) | [Paper](https://www.overleaf.com/read/zktghsywfcvp#79bc7a)

## Instructions

This repo is separated into two parts:

- `partfield-semantic` handles semantic part segmentation. Code is adapted from PartField and uses Qwen2.5-VL to enable semantic querying.

- `local-control-3d-generation` handles 3D object generation. Code is adapted from threestudio

Follow setup and running instructions in each directory

## Demo

Given arbitrary object meshes, one can reimagine them by first running part segmentation and then running generation.
Below is an example on a ScanNet++ scene, imported into Unity
