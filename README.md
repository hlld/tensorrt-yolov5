# tensorrt-yolov5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

This repo contains a slightly modified version of [tensorrtx](https://github.com/wang-xinyu/tensorrtx), and only support yolov5 v3.0.

- For yolov5 v3.0, please visit [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0), and use the latest commit of repo tensorrtx.
- For yolov5 v2.0, please visit [yolov5 release v2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0), and checkout commit ['7cd092d'](https://github.com/wang-xinyu/tensorrtx/commit/7cd092d38289123442157cf7defab78e816f4440) of repo tensorrtx.
- For yolov5 v1.0, please visit [yolov5 release v1.0](https://github.com/ultralytics/yolov5/releases/tag/v1.0), and checkout commit ['0504551'](https://github.com/wang-xinyu/tensorrtx/commit/0504551c0b7d0bac5f998eda349810ba410715de) of repo tensorrtx.

## Config

- Specify the model s/m/l/x when running command ./yolov5
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h
- FP16/FP32 can be selected by the macro in yolov5.cpp
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp, default 1

## How to Run, yolov5s as example

```
1. generate yolov5sv3.wts from pytorch with yolov5sv3.pt

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
// download its weights 'yolov5sv3.pt'
// copy tensorrtx/yolov5/gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5sv3.pt and yolov5sv3.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5sv3.wts' will be generated.

2. build tensorrtx/yolov5 and run

// put yolov5sv3.wts into tensorrtx/yolov5
// go to tensorrtx/yolov5
mkdir build
cd build
cmake ..
make
sudo ./yolov5 -s s           // serialize model to plan file i.e. 'yolov5sv3.engine'
sudo ./yolov5 -e s -d  ../images // deserialize plan file and run inference, the images in samples will be processed.

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

// install python-tensorrt, pycuda, etc.
// ensure the yolov5sv3.engine and libmyplugins.so have been built
python yolov5_trt.py
```

## More Information

See the readme in [tensorrtx home page.](https://github.com/wang-xinyu/tensorrtx)
