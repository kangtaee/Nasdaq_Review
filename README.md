# Nasdaq_Review
About MobileBert를 활용한 nasdaq 뉴스헤드라인 문장 감성분석
# 📊 NASDAQ 뉴스 감성 분석 및 주가 반응 예측

본 프로젝트는 NASDAQ 종목들의 뉴스 헤드라인/요약 내용을 바탕으로 **감성(Sentiment)을 자동 분류**하고,  
분류된 감성과 **실제 주가 흐름(수익률)** 간의 상관관계를 분석합니다.  
이 과정은 자연어 처리 기반의 rule-based 라벨링과 머신러닝 학습 기반 접근 모두를 포함합니다.

---

## 🗂 프로젝트 개요

- 📅 데이터 기간: 2023년 1월
- 📰 뉴스 출처: NASDAQ RSS (헤드라인 및 요약 정보)
- 📈 주가 데이터: NASDAQ 시가/종가 정보
- 🎯 주요 목표:
  - 감성 사전 기반 뉴스 분류 (`긍정`, `중립`, `부정`)
  - 감성과 실제 주가 흐름 간의 일치 여부 분석
  - 다음날 수익률과의 관계 시각화

---

## 🧹 데이터 전처리 과정

뉴스 요약(news_smy_ifo) 컬럼에서 다음과 같은 감성 단어를 기준으로 rule-based 라벨링을 수행합니다:

- 🔵 **Positive keywords**  
  `surge`, `beat`, `gain`, `rally`, `recover`, `record`, `soar`, `strong`, ...

- 🔴 **Negative keywords**  
  `drop`, `fall`, `loss`, `miss`, `weak`, `crisis`, `cut`, `plunge`, ...

> ✔ 긍정 단어만 있으면 `Positive(1)`,  
> ✔ 부정 단어만 있으면 `Negative(2)`,  
> ✔ 둘 다 포함되거나 없으면 `Neutral(0)`

| 원본 뉴스 예시 | 라벨링 결과 |
|----------------|-------------|
| PhenixFIN (PFX) has been beaten down lately ... | Positive |
| There's a disconnect setting up in the energy ... | Neutral |
| Fintel reports that Hoak Public Equities, LP ... | Neutral |

![전처리 예시](img/스크린샷 2025-05-19 103454.png)

---

## 📈 감성 분포 시각화

### ▶ 날짜별 감성 변화

뉴스 발생일 기준 감성 분포의 변화 추이를 나타냅니다.

![감성 시계열](img/스크린샷%202025-05-19%20103526.png)

---

### ▶ 종목별 감성 분포 (Top 10)

뉴스 언급이 많은 상위 종목 기준 감성 분포를 정리한 바 차트입니다.

![종목별 감성](img/스크린샷%202025-05-19%20103540.png)

---

## 📉 감성 vs 주가 수익률 분석

### ▶ 감성별 다음날 수익률 분포

긍정 뉴스 vs 부정 뉴스 이후 종가 수익률 분포를 시각화한 결과입니다.

![수익률 분포](img/스크린샷%202025-05-19%20103613.png)

---

### ▶ 감성 vs 수익률 산점도

각 뉴스 감성에 대해 다음날 수익률의 퍼짐을 보여주는 분포도입니다.

![감성 vs 수익률](img/스크린샷%202025-05-19%20103622.png)

---

## ✅ 주요 결과 요약

| 분석 항목 | 결과 |
|------------|--------|
| 감성 vs 당일 주가 방향 일치율 | 약 51% |
| 감성 vs 다음날 수익률 방향 일치율 | 약 49% |
| 감성과 수익률 상관관계 | 미약한 음의 상관관계 (−0.06 수준) |
| 종목 필터링 시 | 특정 종목은 패턴 존재 (예: SPI) |

---

## 🧠 활용 가능성

- 실시간 뉴스 감성 라벨링 → 투자 판단 참고 지표
- 종목별 뉴스 반응 패턴 학습을 통한 개인화 추천
- 감성 기반 이상 탐지 시스템 구축 등

---

## 📁 주요 파일 구조

```bash
📦 root
 ┣ 📄 README.md
 ┣ 📄 news_sentiment_labeled.csv         # 감성 라벨링 완료된 데이터
 ┣ 📄 train_mobilebert.py               # 모델 학습 코드 (옵션)
 ┣ 📄 evaluate_model.py                 # 정확도 평가 및 confusion matrix
 ┣ 📄 NASDAQ_DT.csv                     # 종가/시가 기반 수익률 정보
 ┣ 📄 Sentiment_Stock_Report.ipynb      # 시각화 보고서
 ┣ 📄 전처리_감성분석_보고서.ipynb       # 전처리 과정 정리
