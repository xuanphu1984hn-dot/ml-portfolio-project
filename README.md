# AI Stock Return Forecasting & Portfolio Optimization

Ứng dụng Machine Learning trong dự báo lợi suất cổ phiếu và xây dựng danh mục đầu tư tối ưu — Nghiên cứu trên nhóm cổ phiếu vốn hóa lớn thuộc S&P 500.

> Đồ án học thuật — không phải hệ thống giao dịch tài chính thực tế. Mọi kết quả đều được tính từ dữ liệu lịch sử thực nghiệm, không khuyến nghị sử dụng cho mục đích đầu tư thực tế.

**Demo trực tuyến:** [https://ml-portfolio-project-o6zvjgra9osvs2sauh7jfn.streamlit.app/](https://ml-portfolio-project-o6zvjgra9osvs2sauh7jfn.streamlit.app/)

---

## Mục tiêu

Xây dựng hệ thống sử dụng dữ liệu lịch sử OHLCV của 25 cổ phiếu large-cap S&P 500 để:

1. Dự báo lợi suất 5 ngày giao dịch tiếp theo (`future_return_5d`) cho từng cổ phiếu.
2. Xếp hạng và lựa chọn nhóm cổ phiếu tiềm năng (Top-K).
3. Xây dựng danh mục đầu tư và backtest chiến lược trên dữ liệu ngoài mẫu (Test set).
4. Đánh giá trung thực khả năng vượt trội của chiến lược Machine Learning so với chiến lược đầu tư thụ động (Buy & Hold).

## Câu hỏi nghiên cứu

- **RQ1:** Các mô hình ML/DL (Random Forest, XGBoost, LSTM) có dự báo lợi suất tốt hơn baseline hay không?
- **RQ2:** Nhóm đặc trưng nào (Price/Return, Technical Indicators, Volume) đóng góp nhiều nhất vào chất lượng dự báo và hiệu quả danh mục?
- **RQ3:** Chiến lược chọn Top-K cổ phiếu dựa trên dự báo mô hình có tạo ra Sharpe Ratio vượt trội so với benchmark hay không?

## Cấu trúc repository

```
ml-portfolio-project/
├── app.py                      # Streamlit dashboard (4 trang)
├── requirements.txt            # Thư viện cần cài đặt
├── data/
│   └── processed/              # Dữ liệu đã xử lý (OHLCV, feature mới nhất mỗi mã)
├── models/                     # Model đã huấn luyện (XGBoost, Random Forest, LSTM) + scaler
├── notebooks/                  # Notebook pipeline đầy đủ (EDA → Feature Engineering → Backtest)
└── README.md
```

## Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Yahoo Finance (`yfinance`) |
| Tần suất | Daily OHLCV |
| Phạm vi thời gian | 2000-01-03 → 2026-08-13 |
| Universe | 25 cổ phiếu large-cap S&P 500 |
| Chất lượng dữ liệu | 100% đầy đủ, 0% thiếu dữ liệu, đồng bộ lịch giao dịch giữa các mã |

Danh sách 25 mã: `AAPL, MSFT, JNJ, PG, KO, XOM, JPM, WMT, HD, DIS, INTC, CSCO, PFE, MRK, VZ, T, IBM, GE, CAT, MMM, MCD, NKE, CVX, UNH, ORCL`

## Phương pháp

### Đặc trưng (17 biến, 3 nhóm)

| Nhóm | Số lượng | Mô tả |
|---|---|---|
| Price/Return | 5 | Lợi suất 1, 5, 10, 20, 60 ngày |
| Technical Indicators | 10 | MA(5/10/20/50/200)_ratio, RSI, MACD_pct, MACD_signal_pct, ATR_pct, Volatility 20 ngày |
| Volume | 2 | Volume change, Volume ratio |

Toàn bộ đặc trưng dạng giá trị tuyệt đối (MA, MACD, ATR) đã được chuẩn hóa sang dạng **tỷ lệ tương đối** để đảm bảo tính dừng (stationary) khi dữ liệu trải dài 26 năm — chi tiết xem báo cáo, Chương 3.

### Chia dữ liệu theo thời gian (không random split)

| Tập | Khoảng thời gian | Vai trò |
|---|---|---|
| Train | 2000–2017 | Huấn luyện mô hình |
| Validation | 2018–2021 | Chọn hyperparameter, chọn thiết kế chiến lược |
| Test | 2022–2026 | Đánh giá cuối cùng, **chỉ sử dụng đúng một lần** |

### Mô hình

Linear Regression (baseline có học) · Random Forest · XGBoost · LSTM — tất cả huấn luyện như bài toán **hồi quy** (dự báo một con số lợi suất kỳ vọng), không phải bài toán phân loại BUY/SELL.

### Pipeline Prediction → Portfolio

```
OHLCV + 17 Features (tại thời điểm T)
        ↓
   Mô hình (hồi quy)
        ↓
Predicted future_return_5d  (một số thực)
        ↓
   Ranking (xếp hạng 25 mã)
        ↓
   Top-K (chọn K mã cao nhất)
        ↓
Portfolio Weighting (Equal Weight / Volatility Weight)
        ↓
   Backtest (rebalance mỗi 5 ngày)
```

> **Lưu ý quan trọng:** Mô hình dự báo **một giá trị lợi suất liên tục**, không trực tiếp tạo ra tín hiệu BUY/HOLD/SELL. Nhãn BUY/HOLD/SELL hiển thị trong dashboard chỉ là quy tắc hậu xử lý dựa trên thứ hạng dự báo, phục vụ mục đích trực quan hóa.

## Kết quả chính (Test set, 2022–2026)

| Chiến lược | Sharpe Ratio | Total Return | CAGR | Annualized Volatility | Max Drawdown | Win Rate | Calmar Ratio |
|---|---|---|---|---|---|---|---|
| **Buy & Hold** | **1.068** | 87.88% | 14.81% | 13.82% | -18.92% | 54.56% | 0.782 |
| Equal Weight | 1.026 | 82.80% | 14.12% | 13.80% | -19.92% | 53.69% | 0.709 |
| XGBoost (Top-5, Equal Weight) | 0.836 | 100.55% | 16.46% | 20.81% | -20.32% | 53.43% | 0.810 |
| LSTM | 0.759 | 90.73% | 15.18% | 21.72% | -20.31% | 51.00% | 0.748 |
| Random Forest | 0.734 | 83.50% | 14.21% | 21.16% | -20.85% | 52.56% | 0.682 |
| Linear Regression | 0.714 | 82.42% | 14.07% | 21.70% | -21.21% | 53.00% | 0.663 |
| Proposed Strategy (Top-5, Volatility Weight) | 0.488 | 41.86% | 7.96% | 19.63% | -26.14% | 53.61% | 0.304 |

**Kết luận:** Mục tiêu Sharpe Ratio ≥ 1.8 đặt ra ban đầu **không đạt được** bằng phương pháp trung thực. Chiến lược tốt nhất là Buy & Hold thụ động — kết quả này phù hợp với lý thuyết thị trường hiệu quả dạng yếu, khi dữ liệu đầu vào chỉ giới hạn ở giá và khối lượng giao dịch công khai. Đáng chú ý, XGBoost (Top-5, Equal Weight) đạt Total Return cao nhất trong toàn bộ thử nghiệm (100.55%) và Sharpe Ratio cao nhất trong nhóm mô hình ML/DL, nhưng vẫn chưa vượt qua benchmark thụ động về hiệu suất điều chỉnh rủi ro.

### Phát hiện quan trọng nhất: Backtest / Validation Overfitting

Proposed Strategy (XGBoost, Top-5, Volatility Weight) đạt Sharpe cao trên tập Validation trong quá trình lựa chọn thiết kế — nhưng khi áp dụng đúng một lần lên Test, Sharpe sụt xuống còn **0.488**. Đây là bằng chứng thực nghiệm rõ ràng rằng chiến lược đã "học" quá khớp đặc thù biến động thị trường của giai đoạn 2018–2021 (bao gồm COVID-19), không tổng quát hóa tốt sang giai đoạn 2022–2026 (chu kỳ tăng lãi suất). Chi tiết phân tích xem báo cáo, Chương 5.

### Ablation Study

| Thí nghiệm | Số đặc trưng | IC | Sharpe |
|---|---|---|---|
| A: Price/Return only | 5 | 0.0134 | 0.584 |
| B: A + Technical Indicators | 15 | 0.0145 | 0.714 |
| C: B + Volume (full) | 17 | 0.0135 | 0.777 |

Technical Indicators đóng góp mạnh nhất vào hiệu suất; Volume features đóng góp nhẹ hơn nhưng vẫn tích cực.

## Hạn chế phương pháp luận đã tự phát hiện qua Audit

Sau khi hoàn thành pipeline, nhóm nghiên cứu tự thực hiện một vòng audit độc lập và phát hiện các hạn chế sau (đã ghi rõ trong báo cáo, không che giấu):

1. **Execution timing bias:** backtest hiện tại dùng giá đóng cửa của cùng ngày T để vừa tính tín hiệu vừa thực hiện giao dịch (same-bar execution), trong khi về lý thuyết cần thực hiện tại T+1. Bias này ảnh hưởng đồng đều lên mọi chiến lược nên không làm sai lệch so sánh tương đối, nhưng có thể khiến Sharpe tuyệt đối được ước lượng lạc quan hơn thực tế.
2. **Test set reuse ở Ablation Study:** Ablation Study được đánh giá trên tập Test (đã dùng một lần cho kết quả chính), thay vì Validation.
3. **Soft test contamination:** kết quả Test của 4 mô hình đã được quan sát trước khi chính thức chọn mô hình nền cho Proposed Strategy.

Đây là các hướng cải thiện ưu tiên hàng đầu nếu đồ án được phát triển tiếp.

## Ứng dụng minh họa (Streamlit Dashboard)

4 trang: **Market Overview** (biểu đồ giá) · **AI Stock Prediction** (dự báo + xếp hạng) · **Recommended Portfolio** (Top-K tùy chỉnh + phân bổ trọng số) · **Backtest** (so sánh 7 chiến lược + Ablation Study).

### Chạy ứng dụng

```bash
git clone https://github.com/<your-username>/ml-portfolio-project.git
cd ml-portfolio-project
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`.

## Công nghệ sử dụng

Python · Pandas · NumPy · Scikit-learn · XGBoost · PyTorch (LSTM) · Plotly · Streamlit

## Giới hạn nghiên cứu

- Chỉ sử dụng dữ liệu OHLCV công khai, không có alternative data (tin tức, sentiment, dữ liệu vĩ mô).
- Backtest chưa tính chi phí giao dịch và trượt giá.
- Phạm vi 25 cổ phiếu, nhỏ hơn chuẩn nghiên cứu thực tế.

## Tài liệu tham khảo trong repo

- `docs/BaoCao_DoAn.docx` — báo cáo đầy đủ, gồm cơ sở lý thuyết, phân tích chi tiết từng chương, và phần audit phương pháp luận.
- `notebooks/` — notebook đầy đủ từ EDA đến Backtest, có thể chạy lại toàn bộ pipeline.

## Giấy phép

Dự án phục vụ mục đích học thuật. Vui lòng không sử dụng cho mục đích đầu tư thực tế.