# S3 - Baseline sau khi merge S1 và S2

## 1. Mục tiêu
Chạy lại baseline sau khi các thay đổi S1 và S2 đã được merge vào main.

## 2. Môi trường thực nghiệm
- Môi trường: Ubuntu trên VMware
- Thiết bị tính toán: CPU
- Python: 3.12.3
- PyTorch: 2.14.1+cpu
- Dữ liệu: PTB-XL, được quản lý bằng DVC
- Git commit trên main: c2116db
- Seed: 42
- Epochs: 2
- Batch size: 16
- Learning rate: 0.001
- MLflow experiment: ecg-superdiagnostic
- MLflow run: 71a17fe6dd1d49b58ca4ff1fee32de57

## 3. Dữ liệu
- Train records: 17,418
- Validation records: 2,183
- Test records: 2,198
- Thứ tự superclass: NORM, MI, STTC, CD, HYP

## 4. Kết quả
- Epoch 1: train loss = 0.7016; validation loss = 0.6774; validation macro-F1 = 0.6119
- Epoch 2: train loss = 0.6343; validation loss = 0.6144; validation macro-F1 = 0.6733
- Test loss: 0.6230
- Test macro-F1: 0.6699
- Test macro-AUC: 0.8882
- Test micro-AUC: 0.9048

### Test F1 theo superclass
| Superclass | F1 |
|---|---:|
| NORM | 0.8414 |
| MI | 0.6838 |
| STTC | 0.7195 |
| CD | 0.7049 |
| HYP | 0.4000 |

### Test AUC theo superclass
| Superclass | AUC |
|---|---:|
| NORM | 0.9305 |
| MI | 0.8933 |
| STTC | 0.9277 |
| CD | 0.8899 |
| HYP | 0.7997 |

## 5. Artifact và minh chứng
- Checkpoint: `artifacts/baseline-s1s2-2ep/best_model.pt`
- Lịch sử huấn luyện: `history.json`
- Chỉ số test: `test_metrics.json`
- Log: `baseline-s1s2-2ep.log`
- MLflow run: `71a17fe6dd1d49b58ca4ff1fee32de57`

## 6. Ghi chú
- Checkpoint cục bộ đã được kiểm tra và đọc thành công.
- MLflow run có trạng thái FINISHED.
- Có cảnh báo về nhãn phiên bản PyTorch `+cpu` khi MLflow ghi requirements.
- Chưa xác nhận model artifact được đăng ký trong MLflow có thể truy cập bình thường.
- Đây là kết quả baseline 2 epoch sau S1 và S2; không so sánh trực tiếp với `baseline-2ep` nếu chưa nêu rõ khác biệt về mã nguồn, cấu hình và môi trường.
- Kết quả phục vụ mục đích học thuật, không dùng để chẩn đoán y tế.
