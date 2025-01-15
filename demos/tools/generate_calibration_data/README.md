# 如何生成校准数据集

## 方法一
使用OpenExplore包每个转换示例的02_preprocess.sh脚本, 生成校准数据集。
OpenExplore包的获取方式参考本仓库总README的**RDK算法工具链资源**小节。

如果遇到类似`Can't reshape 1354752 in (1,3,640,640)`的错误，请修改同级目录下preprocess.py文件中的分辨率，修改为准备转化的onnx一样大小的分辨率，并删除所有的校准数据集，再重新运行02脚本，生成校准数据集。
目前这个示例的校准数据集来自../../../01_common/calibration data/coco目录，生成在./calibration_data_rgb_f32目录。

## 方法二
使用社区博主玺哥文章中介绍的方法, 使用OpenCV + Numpy生成。

## 方法三
使用本目录下的脚本生成校准数据集, 参考了玺哥的方法。

# 参考
[小玺玺: [BPU部署教程] 一文带你轻松走出模型部署新手村](https://developer.d-robotics.cc/forumDetail/107952931390742029)