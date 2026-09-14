import os
import argparse
import subprocess



RED_BEGIN = "\033[1;31m"
GREEN_BEGIN = "\033[1;32m"
COLOR_END = "\033[0m"

def main():
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="", help='Path to Load Test Image.')
    parser.add_argument('--max', type=int, default=2, help='Classes Num to Detect.')
    opt = parser.parse_args()
    # 输出perf信息
    opt.file = os.path.join(os.getcwd(), opt.file)
    print("Perf file: %s"%opt.file)
    print("Max Thread_num: %d"%opt.max)
    
    # 检查是否有 hrt_model_exec 命令
    result = subprocess.run("hrt_model_exec", shell=True, capture_output=True, text=True).stdout
    if "command not found" in result:
        print("hrt_model_exec not found.")
        exit(-1)
    print("hrt_model_exec found, continue.")

    # 检查目标目录是否存在
    if not os.path.exists(opt.file):
        print("File not found.")
        exit(-1)

    # 获取当前的目标目录文件
    file_names = os.listdir(opt.file)

    # TODO: 获取所有*.bin文件对应的文件大小，并排序


    # 输出hrut_somstatus的信息
    print(RED_BEGIN + "hrut_somstatus" + COLOR_END)

    # TODO: 输出所有*.bin文件的名称和大小
    # 开始逐个perf
    for file_name in file_names:
        if not file_name.endswith(".bin"):
            continue
        print(GREEN_BEGIN + "Model: %s"%file_name + COLOR_END)
        for i in range(opt.max):
            result_str = get_perf_data(os.path.join(opt.file, file_name), i+1)
            print(result_str, end=" ", flush=True)
            if i+1 != opt.max:
                print("<br/>", end=" ", flush=True)
        print("\n")
    
    # TODO
    ## 输出perf进度
    ## 输出CPU温度
    # /sys/class/hwmon/hwmon0
    ## 输出perf的耗时和总耗时
    ## 输出perf结果

def get_perf_data(model_file, thread_num):
    cmd = "hrt_model_exec perf --model_file %s --thread_num %s"%(model_file, thread_num)
    # print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=500).stdout
    output_lines = result.splitlines()
    latency, fps = 0, 0
    for line in output_lines:
        if "Average    latency    is:" in line:
            latency = float(line.split("Average    latency    is: ")[-1].split(" ms")[0])
        elif "Frame      rate       is:" in line:
            fps = float(line.split("Frame      rate       is: ")[-1].split(" FPS")[0])
    if thread_num == 1:
        result_str = "%.1f ms / %.1f FPS (%d thread  )"%(latency, fps, thread_num)
    else:
        result_str = "%.1f ms / %.1f FPS (%d threads)"%(latency, fps, thread_num)
    return result_str

if __name__ == "__main__":
    main()
