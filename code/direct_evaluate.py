import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# 1. 평가 데이터 불러오기
df = pd.read_csv("news_sentiment_train_content_isolated.csv", encoding="utf-8")
data_X = list(df['content'].astype(str).values)
labels = df['sentiment'].astype(int).values

# 2. 토크나이저 로드 및 토큰화
tokenizer = MobileBertTokenizer.from_pretrained("direct_mobilebert_finance_news")
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

# 3. 평가용 DataLoader 구성
batch_size = 16
eval_dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels))
eval_loader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batch_size)

# 4. 모델 로드
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)
model.load_state_dict(torch.load("direct_mobilebert_finance_news.pt", map_location=device))
model.to(device)
model.eval()

# 5. 예측 수행
preds, true = [], []

for batch in tqdm(eval_loader, desc="📊 Evaluating"):
    b_input_ids, b_mask, b_labels = [b.to(device) for b in batch]
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

    preds.extend(predictions.cpu().numpy())
    true.extend(b_labels.cpu().numpy())

# 6. 평가 결과 출력
accuracy = np.mean(np.array(preds) == np.array(true))
print(f"\n🎯 평가 정확도: {accuracy:.4f}\n")

print("📌 Classification Report:")
print(classification_report(true, preds, target_names=["중립(0)", "호재(1)", "악재(2)"]))

print("📉 Confusion Matrix:")
print(confusion_matrix(true, preds))
