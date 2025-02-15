import re

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)  # 去除 HTML 标签
    text = re.sub(r"[^a-zA-Z]", " ", text)  # 仅保留字母
    text = text.lower().split()
    return " ".join(text)
