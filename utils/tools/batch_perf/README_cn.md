# BATCH PERF

使用Python脚本, 对某一个特定文件夹内的所有*bin结尾的模型文件进行perf.

顺序: perf的顺序按照模型文件的大小，从小到大.

线程数量: 从1到MAX_NUM, 其中MAX_NUM一般设置到2. 

映射
```bash
alias perf='python3 <path to your rdk_model_zoo>/demos/basic/batch_perf/batch_perf.py'
```
