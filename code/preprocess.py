import pandas as pd

def load_and_clean_csv(path):
    df = pd.read_csv(path, encoding='cp949')
    df = df.dropna(subset=["news_smy_ifo"])
    df['news_smy_ifo'] = df['news_smy_ifo'].astype(str)
    return df

def label_sentiment(text):
    positive = [
        'surge', 'beat', 'record', 'increase', 'gain', 'rise', 'soar', 'strong', 'profit',
        'expand', 'growth', 'outperform', 'bullish', 'upgrade', 'high demand', 'positive outlook'
    ]
    negative = [
        'fall', 'drop', 'decline', 'loss', 'miss', 'weak', 'plunge', 'layoff', 'cut',
        'slowdown', 'downgrade', 'bearish', 'shortfall', 'negative outlook', 'crisis'
    ]
    text = text.lower()
    if any(word in text for word in positive):
        return 1
    elif any(word in text for word in negative):
        return 2
    return 0

def label_to_text(code):
    return {0: "중립", 1: "호재", 2: "악재"}.get(code, "")

def build_dataset(df):
    df['sentiment'] = df['news_smy_ifo'].apply(label_sentiment)
    df['sentiment_label'] = df['sentiment'].apply(label_to_text)
    df_final = df[['news_smy_ifo', 'sentiment', 'sentiment_label']].rename(columns={
        'news_smy_ifo': 'sentence'
    })
    return df_final

if __name__ == "__main__":
    input_path = "NASDAQ_RSS_IFO_202301.csv"
    output_path = "news_sentiment_labeled.csv"

    raw_df = load_and_clean_csv(input_path)
    sampled_df = raw_df.sample(frac=0.1, random_state=42)  # 10% 샘플링
    labeled_df = build_dataset(sampled_df)
    labeled_df.to_csv(output_path, index=False, encoding="cp949")
    print(f"✅ 뉴스 라벨링 완료 (10% 샘플): {output_path} 저장됨")

