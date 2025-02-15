import os
import json
import logging
import time
import torch

# =========================
# 1. 日志记录
# =========================
def setup_logger(name="app_logger", level=logging.INFO):
    """设置日志记录"""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # 防止重复添加
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

# 用法：
# logger = setup_logger()
# logger.info("这是一条普通日志")
# logger.error("这是一条错误日志")

# =========================
# 2. 文件操作
# =========================
def check_file_exists(file_path):
    """检查文件是否存在"""
    return os.path.exists(file_path)

def save_json(data, file_path):
    """保存 JSON 文件"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """加载 JSON 文件"""
    if not check_file_exists(file_path):
        raise FileNotFoundError(f"{file_path} 不存在")
    with open(file_path, 'r') as f:
        return json.load(f)

# 用法：
# save_json({"key": "value"}, "example.json")
# data = load_json("example.json")

# =========================
# 3. GPU/设备检查
# =========================
def get_device():
    """获取设备（优先选择 GPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用法：
# device = get_device()
# print(f"当前设备：{device}")

# =========================
# 4. 时间统计
# =========================
class Timer:
    """用于代码块的时间统计"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] 完成，耗时: {elapsed_time:.2f} 秒")

# 用法：
# with Timer("BERT 嵌入计算"):
#     vector = get_bert_embedding(text)
