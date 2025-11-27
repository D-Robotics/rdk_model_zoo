


detect1



| Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| yolo12n_detect  | 640×640        |        80 | 39.70 ms / 25.17 FPS (1 thread ) <br/> 73.19 ms / 27.24 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| yolo12s_detect  | 640×640        |        80 | 63.74 ms / 15.68 FPS (1 thread ) <br/> 121.24 ms / 16.45 FPS (2 threads) | 5.0 ms                         | 9.3 M       | 21.4 M     |
| yolo12m_detect  | 640×640        |        80 | 103.02 ms / 9.70 FPS (1 thread ) <br/> 199.58 ms / 9.99 FPS (2 threads)  | 5.0 ms                         | 20.2 M      | 67.5 M     |
| yolo12l_detect  | 640×640        |        80 | 183.00 ms / 5.46 FPS (1 thread ) <br/> 359.03 ms / 5.56 FPS (2 threads)  | 5.0 ms                         | 26.4 M      | 88.9 M     |
| yolo12x_detect  | 640×640        |        80 | 315.16 ms / 3.17 FPS (1 thread )                                         | 5.0 ms                         | 59.1 M      | 199.0 M    |
| yolo11n_detect  | 640×640        |        80 | 8.25 ms / 121.05 FPS (1 thread ) <br/> 10.56 ms / 188.57 FPS (2 threads) | 5.0 ms                         | 2.6 M       | 6.5 M      |
| yolo11s_detect  | 640×640        |        80 | 15.81 ms / 63.16 FPS (1 thread ) <br/> 25.74 ms / 77.43 FPS (2 threads)  | 5.0 ms                         | 9.4 M       | 21.5 M     |
| yolo11m_detect  | 640×640        |        80 | 34.68 ms / 28.82 FPS (1 thread ) <br/> 63.30 ms / 31.51 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 68.0 M     |
| yolo11l_detect  | 640×640        |        80 | 45.23 ms / 22.10 FPS (1 thread ) <br/> 84.30 ms / 23.66 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 86.9 M     |
| yolo11x_detect  | 640×640        |        80 | 96.70 ms / 10.34 FPS (1 thread ) <br/> 186.76 ms / 10.68 FPS (2 threads) | 5.0 ms                         | 56.9 M      | 194.9 M    |
| yolov10n_detect | 640×640        |        80 | 8.75 ms / 114.19 FPS (1 thread ) <br/> 11.60 ms / 171.72 FPS (2 threads) | 5.0 ms                         | 2.3 M       | 6.7 M      |
| yolov10s_detect | 640×640        |        80 | 14.84 ms / 67.32 FPS (1 thread ) <br/> 23.85 ms / 83.58 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 21.6 M     |
| yolov10m_detect | 640×640        |        80 | 29.40 ms / 33.99 FPS (1 thread ) <br/> 52.83 ms / 37.75 FPS (2 threads)  | 5.0 ms                         | 15.4 M      | 59.1 M     |
| yolov10b_detect | 640×640        |        80 | 40.14 ms / 24.90 FPS (1 thread ) <br/> 74.20 ms / 26.88 FPS (2 threads)  | 5.0 ms                         | 19.1 M      | 92.0 M     |
| yolov10l_detect | 640×640        |        80 | 49.89 ms / 20.04 FPS (1 thread ) <br/> 93.66 ms / 21.30 FPS (2 threads)  | 5.0 ms                         | 24.4 M      | 120.3 M    |
| yolov10x_detect | 640×640        |        80 | 68.92 ms / 14.51 FPS (1 thread ) <br/> 131.54 ms / 15.16 FPS (2 threads) | 5.0 ms                         | 29.5 M      | 160.4 M    |
| yolov9t_detect  | 640×640        |        80 | 6.97 ms / 143.14 FPS (1 thread ) <br/> 7.96 ms / 250.11 FPS (2 threads)  | 5.0 ms                         | 2.1 M       | 8.2 M      |
| yolov9s_detect  | 640×640        |        80 | 13.00 ms / 76.81 FPS (1 thread ) <br/> 20.16 ms / 98.81 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 26.9 M     |
| yolov9m_detect  | 640×640        |        80 | 32.63 ms / 30.63 FPS (1 thread ) <br/> 59.31 ms / 33.62 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 76.8 M     |
| yolov9c_detect  | 640×640        |        80 | 40.46 ms / 24.71 FPS (1 thread ) <br/> 74.77 ms / 26.67 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 102.7 M    |
| yolov9e_detect  | 640×640        |        80 | 119.80 ms / 8.35 FPS (1 thread ) <br/> 233.08 ms / 8.56 FPS (2 threads)  | 5.0 ms                         | 57.4 M      | 189.5 M    |
| yolov8n_detect  | 640×640        |        80 | 7.00 ms / 142.60 FPS (1 thread ) <br/> 8.06 ms / 246.82 FPS (2 threads)  | 5.0 ms                         | 3.2 M       | 8.7 M      |
| yolov8s_detect  | 640×640        |        80 | 13.63 ms / 73.30 FPS (1 thread ) <br/> 21.38 ms / 93.20 FPS (2 threads)  | 5.0 ms                         | 11.2 M      | 28.6 M     |
| yolov8m_detect  | 640×640        |        80 | 30.74 ms / 32.51 FPS (1 thread ) <br/> 55.51 ms / 35.93 FPS (2 threads)  | 5.0 ms                         | 25.9 M      | 78.9 M     |
| yolov8l_detect  | 640×640        |        80 | 59.51 ms / 16.80 FPS (1 thread ) <br/> 112.80 ms / 17.68 FPS (2 threads) | 5.0 ms                         | 43.7 M      | 165.2 M    |
| yolov8x_detect  | 640×640        |        80 | 92.72 ms / 10.78 FPS (1 thread ) <br/> 178.95 ms / 11.15 FPS (2 threads) | 5.0 ms                         | 68.2 M      | 257.8 M    |
| yolov5nu_detect | 640×640        |        80 | 6.33 ms / 157.59 FPS (1 thread ) <br/> 6.80 ms / 291.89 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| yolov5su_detect | 640×640        |        80 | 12.33 ms / 81.04 FPS (1 thread ) <br/> 18.88 ms / 105.56 FPS (2 threads) | 5.0 ms                         | 9.1 M       | 24.0 M     |
| yolov5mu_detect | 640×640        |        80 | 26.57 ms / 37.62 FPS (1 thread ) <br/> 47.20 ms / 42.24 FPS (2 threads)  | 5.0 ms                         | 25.1 M      | 64.2 M     |
| yolov5lu_detect | 640×640        |        80 | 52.83 ms / 18.92 FPS (1 thread ) <br/> 99.42 ms / 20.06 FPS (2 threads)  | 5.0 ms                         | 53.2 M      | 135.0 M    |
| yolov5xu_detect | 640×640        |        80 | 91.55 ms / 10.92 FPS (1 thread ) <br/> 176.49 ms / 11.30 FPS (2 threads) | 5.0 ms                         | 97.2 M      | 246.4 M    |



detect1



| Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| yolo12n_detect  | 640×640        |        80 | 2.65 ms / 368.54 FPS (1 thread ) <br/> 4.43 ms / 443.33 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| yolo12s_detect  | 640×640        |        80 | 4.48 ms / 220.08 FPS (1 thread ) <br/> 8.10 ms / 244.66 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| yolo12m_detect  | 640×640        |        80 | 9.27 ms / 107.09 FPS (1 thread ) <br/> 17.56 ms / 113.12 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| yolo12l_detect  | 640×640        |        80 | 14.66 ms / 67.85 FPS (1 thread ) <br/> 28.30 ms / 70.28 FPS (2 threads)  | 2.0 ms                         | 26.4 M      | 88.9 M     |
| yolo12x_detect  | 640×640        |        80 | 24.72 ms / 40.33 FPS (1 thread ) <br/> 48.27 ms / 41.26 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| yolo11n_detect  | 640×640        |        80 | 1.62 ms / 596.53 FPS (1 thread ) <br/> 2.39 ms / 813.87 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 6.5 M      |
| yolo11s_detect  | 640×640        |        80 | 2.63 ms / 371.42 FPS (1 thread ) <br/> 4.39 ms / 448.18 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| yolo11m_detect  | 640×640        |        80 | 5.63 ms / 175.69 FPS (1 thread ) <br/> 10.35 ms / 191.62 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 68.0 M     |
| yolo11l_detect  | 640×640        |        80 | 6.96 ms / 142.36 FPS (1 thread ) <br/> 13.02 ms / 152.41 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 86.9 M     |
| yolo11x_detect  | 640×640        |        80 | 13.13 ms / 75.78 FPS (1 thread ) <br/> 25.24 ms / 78.82 FPS (2 threads)  | 2.0 ms                         | 56.9 M      | 194.9 M    |
| yolov10n_detect | 640×640        |        80 | 1.58 ms / 608.94 FPS (1 thread ) <br/> 2.32 ms / 837.04 FPS (2 threads)  | 2.0 ms                         | 2.3 M       | 6.7 M      |
| yolov10s_detect | 640×640        |        80 | 2.53 ms / 385.50 FPS (1 thread ) <br/> 4.18 ms / 471.09 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| yolov10m_detect | 640×640        |        80 | 4.49 ms / 219.98 FPS (1 thread ) <br/> 8.11 ms / 244.17 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| yolov10b_detect | 640×640        |        80 | 6.28 ms / 157.57 FPS (1 thread ) <br/> 11.65 ms / 170.32 FPS (2 threads) | 2.0 ms                         | 19.1 M      | 92.0 M     |
| yolov10l_detect | 640×640        |        80 | 7.95 ms / 124.70 FPS (1 thread ) <br/> 14.98 ms / 132.53 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| yolov10x_detect | 640×640        |        80 | 10.83 ms / 91.79 FPS (1 thread ) <br/> 20.66 ms / 96.17 FPS (2 threads)  | 2.0 ms                         | 29.5 M      | 160.4 M    |
| yolov9t_detect  | 640×640        |        80 | 1.77 ms / 546.03 FPS (1 thread ) <br/> 2.67 ms / 730.68 FPS (2 threads)  | 2.0 ms                         | 2.1 M       | 8.2 M      |
| yolov9s_detect  | 640×640        |        80 | 2.74 ms / 357.91 FPS (1 thread ) <br/> 4.62 ms / 425.97 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| yolov9m_detect  | 640×640        |        80 | 5.52 ms / 179.23 FPS (1 thread ) <br/> 10.13 ms / 195.30 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 76.8 M     |
| yolov9c_detect  | 640×640        |        80 | 6.98 ms / 142.00 FPS (1 thread ) <br/> 13.05 ms / 151.95 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 102.7 M    |
| yolov9e_detect  | 640×640        |        80 | 17.75 ms / 56.15 FPS (1 thread ) <br/> 34.41 ms / 57.85 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| yolov8n_detect  | 640×640        |        80 | 1.53 ms / 632.06 FPS (1 thread ) <br/> 2.24 ms / 868.87 FPS (2 threads)  | 2.0 ms                         | 3.2 M       | 8.7 M      |
| yolov8s_detect  | 640×640        |        80 | 2.63 ms / 371.16 FPS (1 thread ) <br/> 4.41 ms / 446.48 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| yolov8m_detect  | 640×640        |        80 | 5.18 ms / 190.64 FPS (1 thread ) <br/> 9.45 ms / 209.80 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| yolov8l_detect  | 640×640        |        80 | 9.97 ms / 99.68 FPS (1 thread ) <br/> 19.00 ms / 104.65 FPS (2 threads)  | 2.0 ms                         | 43.7 M      | 165.2 M    |
| yolov8x_detect  | 640×640        |        80 | 15.77 ms / 63.15 FPS (1 thread ) <br/> 30.53 ms / 65.20 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| yolov5nu_detect | 640×640        |        80 | 1.42 ms / 674.92 FPS (1 thread ) <br/> 2.02 ms / 959.05 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| yolov5su_detect | 640×640        |        80 | 2.31 ms / 420.83 FPS (1 thread ) <br/> 3.79 ms / 519.22 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| yolov5mu_detect | 640×640        |        80 | 4.50 ms / 218.77 FPS (1 thread ) <br/> 8.11 ms / 244.06 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| yolov5lu_detect | 640×640        |        80 | 8.96 ms / 110.78 FPS (1 thread ) <br/> 16.97 ms / 117.15 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| yolov5xu_detect | 640×640        |        80 | 15.97 ms / 62.32 FPS (1 thread ) <br/> 30.90 ms / 64.41 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |



detect1



| Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| yolo12n_detect  | 640×640        |        80 | 1.88 ms / 513.70 FPS (1 thread ) <br/> 3.07 ms / 634.97 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| yolo12s_detect  | 640×640        |        80 | 3.10 ms / 315.83 FPS (1 thread ) <br/> 5.50 ms / 357.85 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| yolo12m_detect  | 640×640        |        80 | 6.47 ms / 152.80 FPS (1 thread ) <br/> 12.18 ms / 162.62 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| yolo12l_detect  | 640×640        |        80 | 10.23 ms / 97.01 FPS (1 thread ) <br/> 19.67 ms / 101.04 FPS (2 threads) | 2.0 ms                         | 26.4 M      | 88.9 M     |
| yolo12x_detect  | 640×640        |        80 | 17.05 ms / 58.34 FPS (1 thread ) <br/> 33.21 ms / 59.92 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| yolo11n_detect  | 640×640        |        80 | 1.16 ms / 816.50 FPS (1 thread ) <br/> 1.66 ms / 1155.65 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 6.5 M      |
| yolo11s_detect  | 640×640        |        80 | 1.81 ms / 533.50 FPS (1 thread ) <br/> 2.98 ms / 656.31 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| yolo11m_detect  | 640×640        |        80 | 3.90 ms / 252.02 FPS (1 thread ) <br/> 7.10 ms / 278.36 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 68.0 M     |
| yolo11l_detect  | 640×640        |        80 | 4.73 ms / 208.61 FPS (1 thread ) <br/> 8.75 ms / 225.99 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 86.9 M     |
| yolo11x_detect  | 640×640        |        80 | 8.84 ms / 112.05 FPS (1 thread ) <br/> 16.92 ms / 117.39 FPS (2 threads) | 2.0 ms                         | 56.9 M      | 194.9 M    |
| yolov10n_detect | 640×640        |        80 | 1.12 ms / 837.97 FPS (1 thread ) <br/> 1.58 ms / 1211.72 FPS (2 threads) | 2.0 ms                         | 2.3 M       | 6.7 M      |
| yolov10s_detect | 640×640        |        80 | 1.75 ms / 548.80 FPS (1 thread ) <br/> 2.81 ms / 692.74 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| yolov10m_detect | 640×640        |        80 | 3.06 ms / 319.65 FPS (1 thread ) <br/> 5.45 ms / 361.32 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| yolov10b_detect | 640×640        |        80 | 4.30 ms / 228.16 FPS (1 thread ) <br/> 7.85 ms / 250.93 FPS (2 threads)  | 2.0 ms                         | 19.1 M      | 92.0 M     |
| yolov10l_detect | 640×640        |        80 | 5.42 ms / 181.96 FPS (1 thread ) <br/> 10.10 ms / 196.04 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| yolov10x_detect | 640×640        |        80 | 7.33 ms / 135.18 FPS (1 thread ) <br/> 13.90 ms / 142.81 FPS (2 threads) | 2.0 ms                         | 29.5 M      | 160.4 M    |
| yolov9t_detect  | 640×640        |        80 | 1.29 ms / 736.75 FPS (1 thread ) <br/> 1.90 ms / 1013.70 FPS (2 threads) | 2.0 ms                         | 2.1 M       | 8.2 M      |
| yolov9s_detect  | 640×640        |        80 | 1.93 ms / 497.53 FPS (1 thread ) <br/> 3.19 ms / 611.75 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| yolov9m_detect  | 640×640        |        80 | 3.77 ms / 260.19 FPS (1 thread ) <br/> 6.83 ms / 288.82 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 76.8 M     |
| yolov9c_detect  | 640×640        |        80 | 4.76 ms / 206.90 FPS (1 thread ) <br/> 8.77 ms / 225.46 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 102.7 M    |
| yolov9e_detect  | 640×640        |        80 | 12.27 ms / 81.00 FPS (1 thread ) <br/> 23.73 ms / 83.75 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| yolov8n_detect  | 640×640        |        80 | 1.10 ms / 851.31 FPS (1 thread ) <br/> 1.52 ms / 1258.50 FPS (2 threads) | 2.0 ms                         | 3.2 M       | 8.7 M      |
| yolov8s_detect  | 640×640        |        80 | 1.83 ms / 524.95 FPS (1 thread ) <br/> 2.95 ms / 660.43 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| yolov8m_detect  | 640×640        |        80 | 3.43 ms / 285.34 FPS (1 thread ) <br/> 6.14 ms / 320.93 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| yolov8l_detect  | 640×640        |        80 | 6.72 ms / 147.19 FPS (1 thread ) <br/> 12.67 ms / 156.40 FPS (2 threads) | 2.0 ms                         | 43.7 M      | 165.2 M    |
| yolov8x_detect  | 640×640        |        80 | 10.44 ms / 95.08 FPS (1 thread ) <br/> 20.11 ms / 98.81 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| yolov5nu_detect | 640×640        |        80 | 0.99 ms / 954.28 FPS (1 thread ) <br/> 1.34 ms / 1418.24 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 7.7 M      |
| yolov5su_detect | 640×640        |        80 | 1.60 ms / 602.38 FPS (1 thread ) <br/> 2.56 ms / 763.66 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| yolov5mu_detect | 640×640        |        80 | 3.06 ms / 319.05 FPS (1 thread ) <br/> 5.43 ms / 363.38 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| yolov5lu_detect | 640×640        |        80 | 6.04 ms / 163.65 FPS (1 thread ) <br/> 11.36 ms / 174.46 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| yolov5xu_detect | 640×640        |        80 | 10.74 ms / 92.40 FPS (1 thread ) <br/> 20.67 ms / 96.10 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |
