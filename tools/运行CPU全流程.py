"""无需安装包即可从仓库根目录启动 CPU 全流程。"""

import sys
from pathlib import Path

# 直接运行 tools 下脚本时，把仓库根目录加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_isp.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
