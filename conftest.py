"""让 pytest 从仓库根目录导入 optflow 包。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
root_text = str(REPO_ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)
