import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# 2. 데이터 로드
df = pd.read_csv("news_sentiment_eval_content_isolated.csv", encoding="utf-8")
data_X = list(df['content'].astype(str).values)
labels = df['sentiment'].astype(int).values

# 3. 학습/검증 데이터 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data_X, labels, test_size=0.2, stratify=labels, random_state=42
)

# 4. 토크나이저 로드
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)

# 5. 토큰화
train_inputs = tokenizer(train_texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
val_inputs = tokenizer(val_texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

# 6. TensorDataset 구성
batch_size = 8  # 증가
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(train_labels))
val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# 7. 모델 정의
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=3)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_loader)*6)  # epochs 반영

# 8. 학습 루프
epochs = 4  # 증가
for epoch in range(epochs):
    print(f"\n🚀 Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    preds, true_labels = [], []

    for batch in tqdm(val_loader, desc="Validating"):
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

    val_accuracy = np.mean(np.array(preds) == np.array(true_labels))
    print(f"🎯 Validation Accuracy: {val_accuracy:.4f}")

# 9. 모델 저장
model.save_pretrained("direct_mobilebert_finance_news")
tokenizer.save_pretrained("direct_mobilebert_finance_news")
torch.save(model.state_dict(), "direct_mobilebert_finance_news.pt")
print("✅ 모델 저장 완료.")
