import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# 1. 데이터 불러오기
df = pd.read_csv("news_sentiment_labeled.csv", encoding="cp949")

data_X = list(df['sentence'].astype(str).values)
labels = df['sentiment'].astype(int).values  # 🔥 수정한 부분!

# 2. 토크나이저 로드 및 토큰화
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert_finance_news")
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 3. 평가용 DataLoader 구성
batch_size = 4
test_data = torch.utils.data.TensorDataset(input_ids, attention_mask, torch.tensor(labels))
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# 4. 모델 로드
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)
model.load_state_dict(torch.load("mobilebert_finance_news.pt", map_location=device))
model.to(device)
model.eval()

# 5. 예측 수행
test_pred, test_true = [], []

for batch in tqdm(test_dataloader, desc="📊 모델 예측 중"):
    batch_ids, batch_mask, batch_labels = [b.to(device) for b in batch]
    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    preds = torch.argmax(logits, dim=1)
    test_pred.extend(preds.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 6. 정확도 출력
accuracy = np.mean(np.array(test_pred) == np.array(test_true))
print(f"\n✅ 뉴스 감성 분류 정확도: {accuracy:.4f}")
