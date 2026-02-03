# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

project = 'BPU_Sample'
copyright = '2026, xiangshun.zhao'
author = 'xiangshun.zhao'

extensions = [
    # 你已有的 breathe/exhale 等 C 部分扩展照旧保留
    "sphinx.ext.napoleon",   # 解析 Google/NumPy docstring
    "autoapi.extension",     # 关键：Python 静态扫描
]

autoapi_type = "python"

# 让 autosummary 自动生成页面
autosummary_generate = True

# 指向你的 Python 代码根目录（可以多个）
# 建议写绝对路径，避免相对路径踩坑
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parents[3]  # 按你的 docs 层级改
autoapi_dirs = [
    str(REPO_ROOT / "samples"),   # 举例：你的 python 在 samples 下
    str(REPO_ROOT / "utils"),     # 举例：也可以加第二个根目录
]

print("AUTOAPI DIRS:", autoapi_dirs)

# 生成到文档内的哪个路径下（会生成 autoapi）
autoapi_root = "autoapi"

# 递归 & 展示内容选项
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# =====（可选但很实用）过滤不想出现在 API 里的文件/目录 =====
autoapi_ignore = [
    "*__pycache__*",
    "*tests*",
    "*dist*",
]

autoapi_keep_files = True

autoapi_generate_api_docs = True

autoapi_add_toctree_entry = False

# =====（可选）让 AutoAPI 不尝试 import =====
# AutoAPI 默认就是解析源码为主；如果你遇到它去 import 的情况，再加这句：
autoapi_python_use_implicit_namespaces = True

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
