import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModel
import torch

# 데이터 불러오기
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 문단 분리 함수
def split_into_paragraphs(text):
    return [p.strip() for p in text.split('\n') if len(p.strip()) > 10]

# 문단 및 soft label 생성
train_paragraphs = []
soft_labels = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    paragraphs = split_into_paragraphs(row['full_text'])
    label = row['generated']
    soft_label = label / len(paragraphs) if len(paragraphs) > 0 else 0.0
    for p in paragraphs:
        train_paragraphs.append(p)
        soft_labels.append(soft_label)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

# 평균 임베딩 추출 함수
def get_avg_embedding(text_list, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        avg_pool = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(avg_pool)
    return embeddings

# 학습 데이터 임베딩
X = np.array(get_avg_embedding(train_paragraphs))
y = np.array(soft_labels)

# Ridge 회귀 모델 학습
clf = Ridge(alpha=1.0)
clf.fit(X, y)

# 테스트 데이터 임베딩
test_paragraphs = test_df['paragraph_text'].tolist()
X_test = np.array(get_avg_embedding(test_paragraphs))

# 예측 및 결과 저장
probs = clf.predict(X_test)
probs = np.clip(probs, 0, 1)

submission = pd.read_csv("sample_submission.csv")
submission['generated'] = probs
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 생성 완료 (soft labeling + Ridge regression)")
