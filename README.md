# AI art generator using CLIP and VQGAN

This repository contains jupyter notebooks to create images from text using AI.
As a generator we use VQGAN from "Taming transformers for high resolution image synthesis." paper.
For guiding synthesis we use CLIP model from OpenAI.

## Resources

Greate article explaining how VQGAN works. Note: We use stage 1 model
here. No transformer is used. https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/

Video showing what VQGAN can also do: shttps://www.youtube.com/watch?v=o7dqGcLDf0A

Article from OpenAI about CLIP: https://openai.com/blog/clip/
Video explaining CLIP paper: https://www.youtube.com/watch?v=T9XSU0pKX2E
Neural network used in CLIP https://www.youtube.com/watch?v=TrdevFK_am4

## Notebooks

All notebooks in the repository:

 - part1_prepare_env.ipynb -> Notebook for installing dependencies and downloading weights.
 - part2_minimal_example.ipynb -> Notebook with minimal example (with no data agumentation).
 - part3_agumentation.ipynb -> Notebook demonstrating agumentation example.
 - part4_advanced_objective.ipynb -> Notebook with multiple prompts.

## Super resolution

Generating images takes a lot of GPU memory. On RTX A6000 we can generate images that have size of 1024x1024 pixels.
Generating larger images is really difficult. We can however generate images in lower resolution and then use
super resolution AI to upscale images to higher resolution. We can use ISR package (Image Super Resolution) package
which implements superresolution neural networks in Tensorflow.

```
python3 superres.py --help
usage: superres.py [-h] [-t MODEL_TYPE] [-m MODEL_FILEPATH] -i INPUT -o OUTPUT
                   [-f]

optional arguments:
  -h, --help            show this help message and exit
  -t MODEL_TYPE, --model-type MODEL_TYPE
                        Model type (L_RDN, S_RDN or DE_L_RDN)
  -m MODEL_FILEPATH, --model-filepath MODEL_FILEPATH
  -i INPUT, --input INPUT
                        Path to image or folder where input images are stored.
  -o OUTPUT, --output OUTPUT
                        Path to image or folder where upscaled images will be
                        stored.
  -f, --overwrite       Overwrite file at destination.
```

For upscaling from 1024x1024 32GB of RAM is sufficient.

The following pretrained models can be used:

 - L_RDN -> Large RDN model (rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5)
 - S_RDN -> Small RDN model (rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5)
 - DE_L_RDN -> Large RDN noise cancelling, detail enhancing model (rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5)

By default we use DE_L_RDN Model and checkpoint file is in weights directory.

With 32GB of RAM 1024x1024 images can be upscaled to 4096x4096 by running them through DE_L_RDN model and then through S_RDN model.

Have fun!
