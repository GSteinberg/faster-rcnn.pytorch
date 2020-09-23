# Faster R-CNN Specialized for Landmine Detection in Remotely Sensed Images

## Introduction

Using this repository, one can generate predictions for the orthomosaics that were split using [ImageSplitter](https://github.com/GSteinberg/ImageSplitter). If you have not already split your orthomosaics, go to [ImageSplitter](https://github.com/GSteinberg/ImageSplitter) to do so. Please follow the directions below carefully to generate accurate predictions.

## Preparation

First, clone the code
```
git clone https://github.com/GSteinberg/faster-rcnn.pytorch.git
```

Then, create a folder where you will put your split orthomasics:
```
cd faster-rcnn.pytorch && mkdir images
```

and the folder where you will put your pretrained model:
```
cd faster-rcnn.pytorch && mkdir models && mkdir resnet101 && mkdir pascal_voc
```

and lastly, the folder where your generated predictions will be outputted:
```
cd faster-rcnn.pytorch && mkdir output && mkdir csvs
```

### Prerequisites

A more detailed walkthrough of the intallation process can be found at this [blog](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) by [ankur6ue](https://github.com/ankur6ue).

First, make sure your environment is set up with these three main prerequisites. 

* Python 2.7 or 3.6
* Pytorch 1.0 or higher
* CUDA 8.0 or higher

Then, install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooling, ROI_Align and ROI_Crop.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter errors during the compilation, you may need to export the CUDA path to your environment variable.**

## Generating predictions

To generate predictions on your split orthomoasaics, these are the steps you must follow:
1. Move all the images you want to generate predictions on into `images/`
2. Download the pretrained model [here](orb.binghamton.edu) **<- make this a real link**
3. Move the pretrained model, `faster_rcnn_1_50_10067.pth`, to `models/res101/pascal_voc`
4. Run the following command to generate predictions
```
python demo.py --net res101 --checksession 1 --checkepoch 50 \
               --checkpoint 10067 --cuda --load_dir models/ \
               --crop_size 700 --crop_stride 70
```

You will find the generated coordinate predictions in `faster-rcnn.pytorch/output/csvs/`.

## Important Note - Please Read

This is a forked repository of Jianwei Yang's Faster R-CNN. Most of the code is unchanged meaning that Jianwei Yang and Jiasen Lu, as well as many others, wrote most of this code. I, Gabriel Steinberg, wrote a small percentage of the code that specializes it for detection of landmines and other UXOs in remotely sensed images.

Since most of the code is unchanged, problems one may have with installation or running the code may be solved by searching for your problem in the long list of issues in the [original Faster R-CNN repository](https://github.com/jwyang/faster-rcnn.pytorch).

Please note that the original repository is no longer being actively maintained but I will be maintaining this fork for the forseeable future. That means that new issues placed in the [original repository](https://github.com/jwyang/faster-rcnn.pytorch) may remain unanswered by the original developers (other contributers may answer though) but I will attempt to answer any issue placed in this repository.

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
