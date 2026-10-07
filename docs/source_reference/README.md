# BPU Sample 源码说明文档

本目录提供 **BPU Sample 源码 API 参考文档**，覆盖仓库根目录 `samples/` 与 `utils/` 下的 Python 与 C/C++ 实现，以静态 HTML 站点形式交付：

- 普通用户：直接解压浏览已构建的文档包，无需任何构建环境。
- 开发者：本地重建文档并打包。

## 一、查看已构建文档（普通用户）

### 1. 解压文档包

在仓库根目录执行：

```bash
mkdir -p /tmp/bpu_sample_docs_html
tar -xf docs/source_reference/bpu_sample_docs_html.tar.xz -C /tmp/bpu_sample_docs_html
```

压缩包根目录即站点根目录，解压后主要入口：

```text
/tmp/bpu_sample_docs_html/
├── index.html            # 站点首页
├── python/               # Python 导航页（samples 按任务/模型分组、utils）
├── autoapi/              # AutoAPI 生成的 Python API 页面
├── cpp/                  # C/C++ 说明页（链接到 Doxygen 站点）
└── doxygen_site/html/    # Doxygen 生成的 C/C++ API 站点（index.html / files.html）
```

### 2. 浏览文档

- 本地（有图形界面）：用浏览器直接打开 `/tmp/bpu_sample_docs_html/index.html`。
- 命令行方式（本地或远程均适用）：

    ```bash
    python3 -m http.server 8000 --directory /tmp/bpu_sample_docs_html
    ```

    在浏览器中访问 `http://localhost:8000`。

- 远程服务器无图形界面时，先在本地终端做端口转发，再按上述方式访问：

    ```bash
    # 例如 ssh -L 8000:localhost:8000 sunrise@192.168.1.1
    ssh -L 8000:localhost:8000 user@remote_host
    ```

## 二、文档内容

- **Python API**（`python/` + `autoapi/`）：由 sphinx-autoapi 静态扫描仓库根的 `samples/` 与 `utils/` 全部 Python 源码生成，跳过 `__pycache__`、`tests`、`dist` 目录；samples 导航按 **任务（vision / llm / robotics / speech / vla）→ 模型** 两级组织，utils 入口为 `python/utils/`。
- **C/C++ API**（`doxygen_site/`）：由 Doxygen 递归扫描仓库根的 `samples/` 与 `utils/` 下的 C/C++ 源文件（`*.c`、`*.cc`、`*.cpp`、`*.h`、`*.hpp` 等），跳过 `build/`、`CMakeFiles/` 目录；从站点首页或 `cpp/` 页面进入。

## 三、本地构建（开发者）

### 目录结构

```text
docs/source_reference/
├── README.md                        # 本说明
├── bpu_sample_docs_html.tar.xz     # 已构建的 HTML 文档包（发布物）
├── doxygen/
│   └── Doxyfile                     # Doxygen 配置：INPUT=../../../samples/ 与 ../../../utils/，输出 ../sphinx/build/html/doxygen_site
└── sphinx/
    ├── Makefile                     # Sphinx 构建入口（make html）
    ├── make.bat                     # Windows 等价入口
    ├── source/                      # 文档源文件（rst）
    │   ├── conf.py                  # Sphinx 配置：扩展 autoapi.extension、sphinx.ext.napoleon；主题 sphinx_rtd_theme；扫描仓库根 samples/ 与 utils/
    │   ├── index.rst                # 站点首页（C/C++、Python 两个入口）
    │   ├── cpp/doxygen_ref.rst      # C/C++ 入口页，链接到 doxygen_site/html/
    │   ├── python/                  # Python 导航页（samples 导航由 gen_samples_nav.py 生成）
    │   └── autoapi/                 # 构建时由 AutoAPI 生成（构建产物）
    ├── tools/
    │   └── gen_samples_nav.py       # 依据 AutoAPI 输出生成 samples 导航页
    └── build/                       # 构建输出（html / doctrees / doxygen_site，构建产物）
```

### 构建环境

系统依赖 Doxygen（生成 C/C++ 部分）：

```bash
sudo apt install -y doxygen        # Ubuntu/Debian；macOS 可用 brew install doxygen
```

Python 依赖建议安装在虚拟环境中，自行创建与管理：

```bash
python3 -m venv ~/.venvs/bpu-docs
source ~/.venvs/bpu-docs/bin/activate
pip install -U sphinx sphinx-autoapi sphinx-rtd-theme
```

依赖与 `sphinx/source/conf.py` 一致：扩展 `autoapi.extension`（Python 静态扫描）、`sphinx.ext.napoleon`（Sphinx 内置，解析 Google/NumPy 风格 docstring），主题 `sphinx_rtd_theme`。

### 构建步骤

从仓库根目录出发，按以下顺序执行：

```bash
# 1. Sphinx 第一次构建：扫描 Python 源码，生成 AutoAPI 文档源文件
#    输入：仓库根 samples/、utils/（conf.py 的 autoapi_dirs）
#    输出：sphinx/source/autoapi/**（rst）与 sphinx/build/html/
cd docs/source_reference/sphinx
make html

# 2. Doxygen 构建：扫描 C/C++ 源码
#    输入：../../../samples/、../../../utils/（Doxyfile 的 INPUT，相对 doxygen/ 目录）
#    输出：sphinx/build/html/doxygen_site/html/
cd ../doxygen
doxygen Doxyfile

# 3. 生成 Samples 导航页
#    输入：sphinx/source/autoapi/samples/**/index.rst
#    输出：sphinx/source/python/samples/index.rst 及 <task>/<model>/index.rst
cd ../sphinx
python3 tools/gen_samples_nav.py .

# 4. Sphinx 第二次构建：纳入导航页，生成最终 HTML 站点
#    输出：docs/source_reference/sphinx/build/html/index.html
make html
```

### 构建产物与打包

最终站点位于 `docs/source_reference/sphinx/build/html/`，首页为 `index.html`。

打包新构建的文档（从仓库根目录执行；输出到 `/tmp`，不改动仓库内文件）：

```bash
tar -cJf /tmp/bpu_sample_docs_html.tar.xz -C docs/source_reference/sphinx/build/html .
```

包根即站点根（`index.html` 位于包根，与已发布文档包布局一致）。仓库内 `docs/source_reference/bpu_sample_docs_html.tar.xz` 为已发布文档包；新增或更新模型、接口注释后，按上述步骤重新构建并替换该文件。
