import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 데이터 불러오기
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 문단 분리 함수
def split_into_paragraphs(text):
    return [p.strip() for p in text.split('\n') if len(p.strip()) > 10]

train_paragraphs = []
train_labels = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    for p in split_into_paragraphs(row['full_text']):
        train_paragraphs.append(p)
        train_labels.append(row['generated'])

# 모델 및 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

# BERT 임베딩 함수 (배치 처리)
def get_batch_embeddings(text_list, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(emb)
    return embeddings

# 추가 특성: 길이 정보
def get_length_features(texts):
    return np.array([
        [len(t), len(t.split()), t.count(','), t.count('"') + t.count("'")]
        for t in texts
    ])

# 학습 데이터 임베딩 + 길이 특성
X_emb = np.array(get_batch_embeddings(train_paragraphs))
X_len = get_length_features(train_paragraphs)
X = np.concatenate([X_emb, X_len], axis=1)
y = np.array(train_labels)

# 모델 1: LightGBM
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    num_leaves=64,
    class_weight='balanced',
    random_state=42
)
clf_lgb.fit(X, y)

# 모델 2: Logistic Regression (비교용)
clf_logreg = LogisticRegression(max_iter=1000)
clf_logreg.fit(X, y)

# 테스트 데이터 임베딩 + 길이 특성
test_paragraphs = test_df['paragraph_text'].tolist()
X_test_emb = np.array(get_batch_embeddings(test_paragraphs))
X_test_len = get_length_features(test_paragraphs)
X_test = np.concatenate([X_test_emb, X_test_len], axis=1)

# 예측 (LightGBM)
probs_lgb = clf_lgb.predict_proba(X_test)[:, 1]

# 예측 (LogReg)
probs_log = clf_logreg.predict_proba(X_test)[:, 1]

# 앙상블 평균
probs = (probs_lgb + probs_log) / 2

# 제출 파일 저장
submission = pd.read_csv("sample_submission.csv")
submission['generated'] = probs
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 생성 완료 (LGBM + LogReg 앙상블)")
