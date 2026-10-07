# YOLOE 测试数据

`office_desk.jpg` 与 `classes.names` 逐字节复制自 S11 源的 office_desk.jpg / coco_extended.names，固定版本 `380e1a2bf42041af54be6f34935e50197cfadff9`。词表含 4585 行，其 SHA-256 为 `1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3`，已与 X5/S26 词表核对。示例图用于操作演示，未提供标注或 mAP 真值；不是校准集，也不用于宣称泛化精度。

`source_s11_result_figure.jpg`（SHA-256 `c53242d5fb3da45dc21736e12356811d41d4a73642ce45b956f70105c3a39cc3`）与 `source_s26_result_figure.jpg`（SHA-256 `95b1c217eeefcb64b635828e8a073a04fc305bceabc0b740b99ea7e25914ee79`）逐字节复制自固定 S 源交付（rdk_s `380e1a2bf42041af54be6f34935e50197cfadff9`）各自的 `test_data/result.jpg`——见 S11 与 S26。它们是被源 README 嵌入的量化 S 发布插图，均叠加在同一张随附 `office_desk.jpg` 上，为文档说明而保留。浮点路线的预期输出来自实际运行本 sample，精度指标来自评估器。S11 源未记录其插图由哪次运行产生；S26 源图注说明其插图使用已发布量化 S100 26n PF 模型的实测输出。

`result.jpg` 为运行时产生的可视化，不是签入的期望结果。两个 `source_*_result_figure.jpg` 文件是固定的源记录副本，始终与该运行时输出路径相互独立。
