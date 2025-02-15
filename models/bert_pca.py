import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import joblib
import os
import numpy as np

# 配置
MODEL_NAME = "bert-base-uncased"
PCA_MODEL_PATH = "pca_model.pkl"
TARGET_DIM = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(device)

# 加载或训练 PCA
def load_or_train_pca():
    if os.path.exists(PCA_MODEL_PATH):
        print("Load an existing PCA model")
        pca = joblib.load(PCA_MODEL_PATH)
    else:
        print("No PCA file found, start training PCA")
        X_train = np.random.rand(100, 768)
        pca = PCA(n_components=TARGET_DIM)
        pca.fit(X_train)
        joblib.dump(pca, PCA_MODEL_PATH)
    return pca

pca = load_or_train_pca()

# 获取 BERT 嵌入并进行 PCA 降维
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        output = bert_model(**tokens)
    
    embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    embedding_reduced = pca.transform(embedding.reshape(1, -1))
    return embedding_reduced
