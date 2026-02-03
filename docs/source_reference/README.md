# BPU Sample Documentation

本仓库用于发布 **BPU Sample 源码说明文档**，包含 C/C++ 与 Python 相关实现的源码结构、接口与示例说明。

文档以 **静态 HTML 站点** 的形式提供，支持：
- 普通用户 **无需任何编译环境，直接浏览**
- 开发者 **本地构建、更新并打包发布**

---

## 一、普通用户（推荐阅读）

> **如果你只是想查看文档内容，请只看本节即可。**

### 获取文档包

仓库中已提供构建完成的文档压缩包：

```text
bpu_sample_docs_html.tar.xz

```

### 解压文档

在 Linux 下：

```bash
mkdir -p bpu_sample_docs_html && \
tar -xf bpu_sample_docs_html.tar.xz -C bpu_sample_docs_html
```

解压后会得到一个目录，内部结构类似：

```bash
index.html
_static/
python/
cpp/
search.html
...
```

### 浏览文档

- 本地环境（有图形界面）
    - 使用任意浏览器直接打开解压目录中的：

        ```text
        index.html
        ```

    - 即可开始浏览完整文档内容。

- SSH / 远程服务器环境（无图形界面）

    - 在本地终端执行：

        ```bash
        # 例如 ssh -L 8000:localhost:8000 sunrise@192.168.1.1
        ssh -L 8000:localhost:8000 user@remote_host
        ```

    - 在远程服务器上进入文档目录并启动临时 HTTP 服务：

        ```bash
        cd /path/to/docs_html
        python3 -m http.server 8000
        ```

    - 然后在本地浏览器中访问：

        ```bash
        http://localhost:8000
        ```
## 二、开发者（文档构建与发布）

> **本节仅面向需要维护 / 更新文档的开发者。**

### 目录结构说明

```bash
.
├── build_docs.sh                  # 文档构建与打包脚本（推荐入口）
├── bpu_sample_docs_html.tar.xz     # 已构建好的文档包（发布物）
├── doxygen/                        # C/C++ 文档配置
│   └── Doxyfile
└── sphinx/                         # Sphinx 文档工程
    ├── source/                     # 文档源文件（rst）
    ├── build/                      # 构建输出（html / doctrees）
    └── tools/                      # 辅助脚本（导航生成等）
```

### 构建环境要求
-  系统依赖（Linux / Ubuntu 推荐）
    ```bash
    sudo apt update
    sudo apt install -y doxygen
    ```
- Python 虚拟环境（强烈推荐）
    > 默认已经有python的基本环境

    所有 Python 依赖均建议安装在 虚拟环境 中，避免污染系统环境。

    ```bash
    python -m venv ~/.venvs/bpu-docs/
    source  ~/.venvs/bpu-docs/bin/activate
    ```

    安装文档依赖：

    ```bash
    pip install -U \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autoapi \
    breathe \
    sphinxcontrib-napoleon
    ```

### 文档构建流程

仓库已提供 统一构建脚本，推荐直接使用：

```bash
./build_docs.sh
```

该脚本会按以下顺序执行：

- Sphinx 第一次构建

    - 清理旧的 autoapi 与中间产物

    - 扫描 Python 源码，生成 AutoAPI 文档

- Doxygen 构建

    - 生成 C/C++ 源码说明

- 自动生成 Samples 导航

    - 根据 AutoAPI 结果生成 Python Samples 的导航页面

- Sphinx 第二次构建

    - 生成最终 HTML 文档站点

- 文档打包

    - 输出高压缩比的文档包（tar.xz / tar.zst）

### 构建产物说明
- 发布压缩包，如有新的模型新增，注意及时更新
    ```bash
    bpu_sample_docs_html.tar.xz
    ```
