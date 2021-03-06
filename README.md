This is the code for the paper

# Learning to Warp for Style Transfer

### [Project Page](https://github.com/xch-liu/learning-warp-st) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Learning_To_Warp_for_Style_Transfer_CVPR_2021_paper.pdf) | [Poster](https://github.com/xch-liu/learning-warp-st) | [Video](https://github.com/xch-liu/learning-warp-st)

<p align='center'>
  <img src='images/teaser.jpg' height="140px">
</ p>
  
  Our method performs non-parametric warping to match artistic geometric style. The above shows content, style (geometry+texture), and output images for a Picasso style transfer (left) and a Salvaor Dali style transfer (right).


https://user-images.githubusercontent.com/58788896/124906358-e30dd380-dfde-11eb-99b2-b7b640a6a8d9.mov



If you find this code useful for your research, please cite
```
@InProceedings{Liu21LWST, 
  author={Xiao-Chang Liu and Yong-Liang Yang and Peter Hall},
  title={Learning to Warp for Style Transfer},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Preresquisites

Dependencies:
* Geometric Warping: [VLFeat](http://www.vlfeat.org/) and [MatConvNet](http://www.vlfeat.org/matconvnet/)
* Texture Rendering: [PyTorch](http://pytorch.org/), [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn)

Pre-trained Models:
* Download the [model](https://drive.google.com/uc?id=1PJJQ0KG2JYfJZDkU4ZOePndKJw63d7Yr&export=download) for geometric warping
 ```
 cd geometric_warping
 mkdir model
 ```
* Download the model for texture rendering
 ```
 cd texture_rendering
 python models/download_model.py
 ```

## Usage

### 1. Run geometric style transfer to warp the content image:
```
cd geometric_warping
run geo_warping.m [--STYLE_IMAGE] [--CONTENT_IMAGE]
```

After warping, empty background regions (if appear) are inpainted with pixels nearby.

### 2. Run texture style transfer to render the warped image:
```
cd texture_rendering
run multi_scale_st.sh [--STYLE_IMAGE] [--CONTENT_IMAGE] [--STYLE_WEIGHT]
```
