# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603700
- **Họ và tên:** Nguyễn Duy Mạnh
- **SEED:** 8400

## 1. Bằng chứng


| Tệp                                                                      | Có? |
| --------------------------------------------------------------------------- | :----: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  |  ☑  |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra |  ☑  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước        |  ☑  |

## 2. Kết quả chính


| Mô hình                       | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) |   Tin/s   | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :-------------------------------- | :---------: | :------: | :---------: | :-----------: | :---------: | :---------: | :----------: | :----------: | :------------: |
| 1. BoW + LogReg (tự code)      |   0.988   | 0.787 |   0.876   |    0.857    |   7.391   | 12501.070 |   0.046   |   0.088   |    0.000    |
| 2. BoW + MLP (deep learning)    |   0.990   | 0.889 |   0.937   |    0.947    |   9.807   | 37930.341 |   0.250   |   0.329   |    0.000    |
| 3a. Encoder (MiniLM) + kNN      |   0.870   | 0.926 |   0.897   |    0.947    |   2.942   |  121.466  |   22.353   |   44.584   |    0.000    |
| 3b. Encoder (MiniLM) + LogReg   |   0.990   | 0.944 |   0.967   |    1.000    |   3.433   | 1324.085 |   6.793   |   8.117   |    0.000    |
| 3c. Encoder (MiniLM) + MLP head |   0.981   | 0.944 |   0.962   |    1.000    |   4.094   | 1694.248 |   6.754   |   9.113   |    0.000    |
| 4a. LLM nhỏ zero-shot          |   0.952   | 1.000 |   0.976   |    0.976    |   0.000   |   0.950   |  969.174  |  1466.593  |    0.017    |
| 4b. LLM lớn zero-shot          |   0.952   | 1.000 |   0.976   |    0.976    |   0.000   |   0.950   |  1453.760  |  2199.890  |    0.043    |
| 4c. LLM nhỏ few-shot           |   1.000   | 0.600 |   0.750   |    0.750    |   0.000   |   1.195   |  879.399  |  1398.396  |    0.051    |

## 3. Dự đoán và thực tế

Ba dự đoán có sự chênh lệch đáng chú ý nhất so với kết quả thực tế:

1. **Dự đoán `llm_fewshot` (Few-shot LLM):**

   - *Dự đoán:* Few-shot sẽ giúp mô hình định hướng chuẩn xác và giữ F1 ở mức rất cao (~0.97), đồng thời chi phí tăng nhẹ do độ dài prompt tăng.
   - *Thực tế:* F1 thực tế tụt mạnh xuống **0.750** (Recall giảm sâu còn **0.600**).
   - *Vì sao lệch:* Trong các ví dụ huấn luyện được chọn ngẫu nhiên cho few-shot có các tin nhắn spam chứa từ ngữ thô tục / người lớn nhạy cảm. Điều này vô tình kích hoạt bộ lọc kiểm duyệt nội dung an toàn (Safety Filter / Content Moderation) của Gemini API, khiến mô hình từ chối phản hồi (trả về rỗng), kéo tụt Recall và F1.
2. **Dự đoán `emb_search` (Tìm kiếm tương đồng với Embedding):**

   - *Dự đoán:* Truy vấn về kỳ nghỉ miễn phí (`complimentary holiday`) sẽ chỉ trả về các tin nhắn rác (spam lừa đảo trúng thưởng).
   - *Thực tế:* Trong top 3 kết quả tương đồng nhất vẫn xuất hiện cả tin nhắn thông thường (`ham`) của bạn bè trao đổi về việc đi nghỉ mát.
   - *Vì sao lệch:* Không gian vector embedding mã hóa sự tương đồng về mặt chủ đề ngữ nghĩa (kỳ nghỉ, du lịch) thay vì phân biệt mục đích / ý đồ xấu (intent). Muốn phân biệt chính xác spam/ham cần phải có thêm lớp phân loại có giám sát (classifier) thay vì chỉ đo khoảng cách vector đơn thuần.
3. **Dự đoán `hard_eval` (Đánh giá trên 50 case khó):**

   - *Dự đoán:* F1 của toàn bộ các mô hình sẽ tụt xuống khoảng 0.5 – 0.7; LLM sẽ bị qua mặt bởi kỹ thuật Prompt Injection.
   - *Thực tế:* Các mô hình BoW/MLP tụt rất sâu (F1 chỉ còn 0.286 – 0.324, Accuracy 50%). Ngược lại, LLM zero-shot thể hiện độ bền bỉ rất cao với Accuracy đạt 98%, và bản LLM có prompt phòng thủ (4d) đạt tuyệt đối **100% Accuracy (F1 = 1.000)**.
   - *Vì sao lệch:* LLM có khả năng phân tích ngữ cảnh bậc cao và hiểu ranh giới phân tách dữ liệu khi được đóng gói trong thẻ phòng thủ `<message>`, vô hiệu hóa hoàn toàn ý đồ tiêm lệnh (injection). Trong khi đó, BoW hoàn toàn bất lực trước kỹ thuật ngụy trang chữ (obfuscation) và spam giọng tự nhiên do giới hạn vốn từ (vocabulary).

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** **1. BoW + LogReg** (F1 từ 0.857 tụt xuống 0.286, Accuracy 50%) và **2. BoW + MLP** (F1 từ 0.947 tụt xuống 0.324, Accuracy 50%).
- **Nhóm case khiến mô hình tụt nhiều nhất:**

  - *Spam giọng tự nhiên (natural_spam):* Cả hai mô hình BoW đều đoán đúng **0%** (0/10 case).
  - *Né từ khoá (obfuscation):* BoW + LogReg đúng 17%, BoW + MLP đúng **0%**.
  - *Thao túng / lừa đảo (manipulation):* BoW + LogReg đúng 12%, BoW + MLP đúng 25%.
- **Hai case cụ thể và cơ chế gây sai:**

  1. **Case Né từ khoá (ID 13):** Tin nhắn spam cố tình bóp méo chữ cái (`w!n`, `cl!ck`, `fr33`, chèn ký tự lạ giữa các từ).
     - *Cơ chế sai:* Bộ tiền xử lý của BoW phụ thuộc vào từ điển cố định xây dựng từ tập train. Khi từ ngữ bị ngụy trang, chúng biến thành các từ nằm ngoài từ điển (Out-Of-Vocabulary - OOV) hoặc các token rác vô nghĩa không mang trọng số spam. Do đó vector đặc trưng toàn số 0, khiến mô hình tính xác suất spam thấp và gán nhãn sai thành ham.
  2. **Case Spam giọng tự nhiên (ID 19):** Tin nhắn lừa đảo nhưng dùng giọng điệu thân mật, tự nhiên như bạn bè nhắn tin rủ rê click link.
     - *Cơ chế sai:* Tin nhắn hoàn toàn không chứa các từ khóa spam kinh điển (như `claim`, `urgent`, `prize`, `call free`). BoW chỉ đếm tần suất xuất hiện của từ đơn lẻ nên thấy toàn từ vựng phổ thông thường ngày, dẫn đến điểm logit thiên hoàn toàn về ham (False Negative).

## 5. Thử thách cá nhân

- **Đề bài phân công:** Thử thách #1 (ứng với `SEED = 8400`, `8400 % 6 = 0`):
  *Đổi ngưỡng quyết định $0.5 \to 0.2$ và $\to 0.8$ cho mô hình Phần 1 (BoW + LogReg). Báo precision / recall / F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?*
- **Cách thực hiện:**
  Sử dụng xác suất dự đoán $P(\text{spam})$ từ hàm `predict_proba(X_test, w, b)`. Gán nhãn dự đoán theo điều kiện $\hat{y} = (P \ge \text{threshold})$ với 3 mức ngưỡng: 0.2, 0.5, 0.8. Sau đó tính các chỉ số phân loại bằng hàm `metrics(y_test, y_pred)`.
- **Kết quả thực nghiệm:**

  - **Ngưỡng 0.2:** Precision = **0.890** | Recall = **0.898** | F1 = **0.894**
  - **Ngưỡng 0.5:** Precision = **0.988** | Recall = **0.787** | F1 = **0.876**
  - **Ngưỡng 0.8:** Precision = **1.000** | Recall = **0.546** | F1 = **0.707**
- **Giải thích & Đề xuất cho Ngân hàng:**

  - Đối với dịch vụ ngân hàng, thiệt hại lớn nhất là lỗi **Dương tính giả (False Positive - FP)**: tin nhắn chứa mã OTP hợp lệ bị hệ thống gắn nhãn spam và chặn lại, khiến khách hàng không thể thực hiện giao dịch tài chính hoặc đăng nhập tài khoản.
  - Do đó, ngân hàng bắt buộc phải ưu tiên tối đa chỉ số **Precision** (hạn chế tối đa FP).
  - Tại **ngưỡng 0.8**, Precision đạt mức tuyệt đối **1.000 (100%)**, đồng nghĩa với việc không có bất kỳ tin nhắn ham nào bị chặn nhầm. Dù Recall giảm còn 0.546 (chấp nhận để lọt một tỷ lệ tin rác), việc đảm bảo 100% tin nhắn giao dịch/OTP đến được tay khách hàng là yêu cầu sống còn. Do đó, ngân hàng **bắt buộc nên chọn ngưỡng 0.8**.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu hệ thống:** Xử lý 10 triệu tin nhắn/ngày, đảm bảo độ trễ thời gian thực $p95 < 50\text{ ms/tin}$.
- **Mô hình đề xuất:** **3b. Encoder (MiniLM) + Logistic Regression**.
- **Dẫn chứng số liệu thực nghiệm:**
  1. *Độ trễ p95:* Đạt **8.117 ms/tin**, hoàn toàn thoả mãn và vượt xa yêu cầu $p95 < 50\text{ ms}$ (nhanh gấp ~6 lần giới hạn SLA cho phép). Trong khi đó, các mô hình LLM (4a/4b/4c) có độ trễ $p95$ từ **1,400 ms đến 2,200 ms**, hoàn toàn không khả thi để chạy trực tiếp trên đường truyền viễn thông thời gian thực.
  2. *Chất lượng phân loại:* Đạt **F1 = 0.967** trên tập test chuẩn (và đạt 1.000 trên tập SUB 40), vượt trội rõ rệt so với mô hình BoW truyền thống (F1 = 0.876).
  3. *Chi phí hạ tầng:* **$0 chi phí API**, có thể tự host và triển khai trực tiếp trên cụm máy chủ nội bộ (on-premise / private cloud). Nếu dùng LLM gọi API cho 10 triệu tin/ngày, chi phí ước tính là $10,000 \times \$0.0173 \approx \$173\text{/ngày}$ (hơn **\$63,000/năm**), gây lãng phí tài nguyên khổng lồ.
- **Đánh đổi chấp nhận:**
  Mô hình 3b có độ chính xác trên các case lừa đảo tinh vi thấp hơn LLM (đạt 60% so với 98%–100% của LLM trên 50 case khó). Để khắc phục điểm yếu này, kiến trúc thực tế có thể triển khai dạng **Cascade Filtering**: Tầng 1 sử dụng 3b để xử lý 99% lưu lượng với tốc độ cực nhanh; chỉ chuyển các tin nhắn có xác suất nằm trong vùng mập mờ ($0.4 \le P \le 0.7$) sang Tầng 2 (LLM hoặc bộ quy tắc chuyên biệt) để thẩm định sâu.

## 7. AI log và điều đã học

### Nhật ký sử dụng AI


| Lần | Bạn hỏi gì                                                                                                  | AI trả lời gì                                                                                                                                              | Bạn sửa/áp dụng gì                                                                                                                                               |
| :----: | :--------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | Bị lỗi `NameError: name 're' is not defined` trong hàm `parse_label` sau khi khởi động lại runtime.     | Giải thích rằng sau khi restart runtime các biến/module chưa được nạp lại, và lệnh `import re` ở đầu notebook chưa chạy.                    | Tôi đã thêm trực tiếp dòng `import re` vào bên trong thân hàm `parse_label` để hàm hoạt động độc lập, không phụ thuộc trạng thái toàn cục. |
|  2  | Bị lỗi `RateLimitError: 429 Resource Exhausted` khi gọi batch Gemini API.                                   | Chỉ ra hạn mức tốc độ của gói Free Tier là 15 RPM (yêu cầu/phút), khuyên nên thêm độ trễ giữa các lượt gọi.                            | Tôi đã thêm `time.sleep(4)` vào vòng lặp của hàm `llm_classify_batch` để giãn cách các request, chạy ổn định không bị lỗi giới hạn.            |
|  3  | Hàm `call_openai` bị lỗi `AttributeError: 'NoneType' object has no attribute 'content'` khi chạy Few-shot. | Giải thích do tin nhắn spam chứa từ khóa nhạy cảm kích hoạt bộ lọc kiểm duyệt (Safety Filter) của Gemini khiến API trả về phản hồi rỗng. | Tôi sửa lại câu lệnh trích xuất tin nhắn sang dạng phòng thủ an toàn: `getattr(resp.choices[0], "message", None)` để tránh lỗi sập chương trình.  |

### Điều tâm đắc học được

Trước buổi lab hôm nay, tôi từng nghĩ rằng các mô hình ngôn ngữ lớn (LLM) hiện đại luôn là giải pháp vượt trội nhất trong mọi trường hợp. Tuy nhiên, qua quá trình trực tiếp đo đạc các chỉ số, tôi nhận ra trong các bài toán sản phẩm thực tế, sự đánh đổi giữa **chất lượng, tốc độ (latency)** và **chi phí vận hành** mới là yếu tố quyết định. Sự kết hợp giữa vector biểu diễn ngữ nghĩa (Embedding) và bộ phân loại tuyến tính cổ điển (Logistic Regression) mang lại hiệu năng phi thường với độ trễ chỉ vài mili-giây và chi phí gần như bằng 0. Ngoài ra, việc hiểu rõ bản chất của bài toán nghiệp vụ (như bảo vệ tin OTP của ngân hàng) để lựa chọn ngưỡng quyết định phù hợp quan trọng không kém việc tối ưu điểm số F1 tổng thể.
