import pandas as pd

# 1. 데이터 로드
sentiment_df = pd.read_csv("news_sentiment_train_content_isolated.csv", encoding="utf-8-sig")
nasdaq_df = pd.read_csv("NASDAQ_DT.csv", encoding="cp949")

# 2. 감성 데이터 정리
sentiment_df = sentiment_df[['date', 'ticker', 'sentiment']]
sentiment_df['date'] = sentiment_df['date'].astype(int)

# 3. 주가 당일 수익률 계산 (종가 - 시가) / 시가
nasdaq_df['change'] = (nasdaq_df['gts_iem_end_pr'] - nasdaq_df['gts_iem_ong_pr']) / nasdaq_df['gts_iem_ong_pr']
nasdaq_df['change_direction'] = nasdaq_df['change'].apply(
    lambda x: 1 if x > 0 else (2 if x < 0 else 0)
)

# 4. 다음날 종가 및 수익률 계산
nasdaq_df = nasdaq_df.sort_values(by=['tck_iem_cd', 'trd_dt'])
nasdaq_df['next_close'] = nasdaq_df.groupby('tck_iem_cd')['gts_iem_end_pr'].shift(-1)
nasdaq_df['next_date'] = nasdaq_df.groupby('tck_iem_cd')['trd_dt'].shift(-1)
nasdaq_df['next_return'] = (nasdaq_df['next_close'] - nasdaq_df['gts_iem_end_pr']) / nasdaq_df['gts_iem_end_pr']
nasdaq_df['next_return_dir'] = nasdaq_df['next_return'].apply(
    lambda x: 1 if x > 0 else (2 if x < 0 else 0)
)

# 5. 감성 데이터와 병합
merged_df = pd.merge(
    sentiment_df, nasdaq_df,
    left_on=["date", "ticker"], right_on=["trd_dt", "tck_iem_cd"],
    how="inner"
)

# 6. 당일 방향 일치 여부 계산
def match_today(row):
    if row['sentiment'] == 1 and row['change_direction'] == 1:
        return 1
    elif row['sentiment'] == 2 and row['change_direction'] == 2:
        return 1
    return 0

# 7. 다음날 방향 일치 여부 계산
def match_nextday(row):
    if row['sentiment'] == 1 and row['next_return_dir'] == 1:
        return 1
    elif row['sentiment'] == 2 and row['next_return_dir'] == 2:
        return 1
    return 0

# 8. 평가
non_neutral = merged_df[merged_df['sentiment'] != 0].copy()
non_neutral['match_today'] = non_neutral.apply(match_today, axis=1)
non_neutral['match_nextday'] = non_neutral.apply(match_nextday, axis=1)

# 9. 정확도 계산
today_acc = non_neutral['match_today'].mean()
nextday_acc = non_neutral['match_nextday'].mean()

# 10. 출력
print(f"✅ 감성 vs 당일 주가 방향 정확도: {today_acc:.4f} ({today_acc*100:.2f}%)")
print(f"✅ 감성 vs 다음날 수익률 방향 정확도: {nextday_acc:.4f} ({nextday_acc*100:.2f}%)")
