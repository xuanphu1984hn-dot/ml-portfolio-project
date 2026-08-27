# AI-Powered Stock Return Forecasting & Portfolio Optimization

Ung dung Machine Learning trong du bao loi suat co phieu va xay dung danh muc dau tu toi uu — Nghien cuu tren nhom co phieu von hoa lon thuoc S&P 500.

## Cau truc project

```
ml-portfolio-project/
├── app.py                      # Streamlit dashboard (4 trang)
├── requirements.txt
├── data/processed/             # Dataset da xu ly (feature + target, da chia train/val/test)
├── models/                     # Model da train (XGBoost, Random Forest, LSTM) + scaler
├── notebooks/                  # Notebook pipeline day du (EDA -> Backtest)
└── README.md
```

## Kết quả chính (Test set, 2022-2026)
| Chiến lược | Sharpe Ratio | Total Return | Max Drawdown |
|---|---|---|---|
| Buy & Hold | 1.068 | 87.88% | -18.92% |
| Equal Weight | 1.026 | 82.80% | -19.92% |
| XGBoost (Top5, Equal Weight) | 0.836 | 100.55% | -20.32% |
| LSTM | 0.759 | 90.73% | -20.31% |
| Random Forest | 0.734 | 83.50% | -20.85% |
| Linear Regression | 0.714 | 82.42% | -21.21% |
| Proposed Strategy (Top5, Volatility Weight) | 0.488 | 41.86% | -26.14% |

Mục tiêu Sharpe >= 1.8 KHÔNG đạt được bằng phương pháp trung thực — xem phân tích nguyên nhân chi tiết trong báo cáo (Phase 12).
## Cai dat va chay

```bash
pip install -r requirements.txt
streamlit run app.py
```
