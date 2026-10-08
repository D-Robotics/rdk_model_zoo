# 如何生成校准数据集

## 方法一
使用 OpenExplore 包中转换示例的 `02_preprocess.sh` 脚本生成校准数据集。
OpenExplore 获取方式见[工具链开发文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)。

如果遇到类似`Can't reshape 1354752 in (1,3,640,640)`的错误，请修改同级目录下preprocess.py文件中的分辨率，修改为准备转化的onnx一样大小的分辨率，并删除所有的校准数据集，再重新运行02脚本，生成校准数据集。
目前这个示例的校准数据集来自../../../01_common/calibration data/coco目录，生成在./calibration_data_rgb_f32目录。

## 方法二
使用OpenCV和numpy等库准备校准数据，校准数据除了yaml中配置的减通道均值和归一化的操作，剩下的全部和训练对齐。
