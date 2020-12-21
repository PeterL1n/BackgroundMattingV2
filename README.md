# Real-Time High-Resolution Background Matting

![Teaser](https://github.com/PeterL1n/Matting-PyTorch/blob/master/images/teaser.gif?raw=true)

Official repository for the paper [Real-Time High-Resolution Background Matting](https://arxiv.org/abs/2012.07810). Our model requires capturing an additional background image and produces state-of-the-art matting results at 4K 30fps and HD 60fps on an Nvidia RTX 2080 TI GPU.

* [Visit project site](https://grail.cs.washington.edu/projects/background-matting-v2/)
* [Watch project video](https://www.youtube.com/watch?v=oMfPTeYDF9g)

&nbsp;

## Overview
* [Updates] (#updates)
* [Download](#download)
    * [Model / Weights](#model--weights)
    * [Video / Image Examples](#video--image-examples)
    * [Datasets](#datasets)
* [Demo](#demo)
    * [Scripts](#scripts)
    * [Notebooks](#notebooks)
* [Usage / Documentation](#usage--documentation)
* [Training](#training)
* [Project members](#project-members)
* [License](#license)
* [Community Projects](#community-projects)

&nbsp;

## Updates

* [Dec 21 2020] We updated our project to MIT License, which permits commercial use!

&nbsp;

## Download

### Model / Weights

* [Download model / weights](https://drive.google.com/drive/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh?usp=sharing)

### Video / Image Examples

* [HD videos](https://drive.google.com/drive/folders/1j3BMrRFhFpfzJAe6P2WDtfanoeSCLPiq) (by [Sengupta et al.](https://github.com/senguptaumd/Background-Matting)) (Our model is more robust on HD footage)
* [4K videos and images](https://drive.google.com/drive/folders/16H6Vz3294J-DEzauw06j4IUARRqYGgRD?usp=sharing)


### Datasets

* VideoMatte240K (Coming soon)
* PhotoMatte85 (Coming soon)

&nbsp;

## Demo

#### Scripts

We provide several scripts in this repo for you to experiment with our model. More detailed instructions are included in the files.
* `inference_images.py`: Perform matting on a directory of images.
* `inference_video.py`: Perform matting on a video.
* `inference_webcam.py`: An interactive matting demo using your webcam.

#### Notebooks
Additionally, you can try our notebooks in Google Colab for performing matting on images and videos.

* [Image matting (Colab)](https://colab.research.google.com/drive/1cTxFq1YuoJ5QPqaTcnskwlHDolnjBkB9?usp=sharing)
* [Video matting (Colab)](https://colab.research.google.com/drive/1Y9zWfULc8-DDTSsCH-pX6Utw8skiJG5s?usp=sharing)

#### Virtual Camera
We provide a demo application that pipes webcam video through our model and outputs to a virtual camera. The script only works on Linux system and can be used in Zoom meetings. For more information, checkout:
* [Webcam plugin](https://github.com/andreyryabtsev/BGMv2-webcam-plugin-linux)

#### Web Demo
Developers in the community has helped us build a web demo. See [Community Projects](#community-projects) section below. 

&nbsp;

## Usage / Documentation

You can run our model using **PyTorch**, **TorchScript**, **TensorFlow**, and **ONNX**. For detail about using our model, please check out the [Usage / Documentation](doc/model_usage.md) page.

&nbsp;

## Training

Training code will be released upon acceptance of the paper.

&nbsp;

## Project members
* [Shanchuan Lin](https://www.linkedin.com/in/shanchuanlin/)*, University of Washington
* [Andrey Ryabtsev](http://andreyryabtsev.com/)*, University of Washington
* [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/), University of Washington
* [Brian Curless](https://homes.cs.washington.edu/~curless/), University of Washington
* [Steve Seitz](https://homes.cs.washington.edu/~seitz/), University of Washington
* [Ira Kemelmacher-Shlizerman](https://sites.google.com/view/irakemelmacher/), University of Washington

<sup>* Equal contribution.</sup>

&nbsp;

## License ##
This work is licensed under the MIT License. If you use our work in your project, we would love you to include an acknowledgement and fill out our [survey](https://docs.google.com/forms/d/e/1FAIpQLSdR9Yhu9V1QE3pN_LvZJJyDaEpJD2cscOOqMz8N732eLDf42A/viewform?usp=sf_link).

&nbsp;

## Community Projects

A list of projects built by third-party developers in the community. If you have a project to share, fill out this [survey]([survey](https://docs.google.com/forms/d/e/1FAIpQLSdR9Yhu9V1QE3pN_LvZJJyDaEpJD2cscOOqMz8N732eLDf42A/viewform?usp=sf_link)).

* [Web Demo by Gradio](https://gradio.app/g/BackgroundMattingV2): Matting your own images from your browser.