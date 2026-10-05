# Cài đặt môi trường và tái tạo bước khảo sát dữ liệu

## 1. Môi trường mục tiêu

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Ubuntu Server 24.04 LTS (amd64) |
| Python | 3.12 (bản `python3` của Ubuntu 24.04), không cài thêm bản khác |
| Docker | Docker Engine cài từ apt repository chính thức của Docker (xem mục 3) |
| Ổ đĩa trống | Tối thiểu khoảng 15 GB (dữ liệu PTB-XL giải nén khoảng 3,0 GB, chưa kể môi trường Python và image Docker) |
| RAM | Khuyến nghị từ 4 GB |

Nếu chạy trong máy ảo, không cần cấu hình GPU: việc huấn luyện mô hình chạy ngoài máy này
(Colab/Kaggle/máy cá nhân); máy này chỉ dùng cho phát triển, serving và hạ tầng.

## 2. Gói hệ thống

```bash
sudo apt update
sudo apt full-upgrade -y
sudo timedatectl set-timezone Asia/Ho_Chi_Minh
sudo apt install -y git curl wget ca-certificates python3-venv
df -h /
```

Nếu `full-upgrade` cập nhật kernel, chạy `sudo reboot` rồi đăng nhập lại. Lệnh `df -h /` để
kiểm tra còn đủ chỗ trống (xem mục 1).

## 3. Docker Engine

Các lệnh dưới đây lấy từ tài liệu chính thức: <https://docs.docker.com/engine/install/ubuntu/>
(mục "Install using the apt repository") và
<https://docs.docker.com/engine/install/linux-postinstall/>. Nếu hai trang này khác với tài
liệu này, làm theo trang chính thức.

Gỡ gói không chính thức nếu có (apt có thể báo không có gói nào, bình thường). Không cài Docker
bằng snap hoặc `docker.io` của Ubuntu vì sẽ xung đột.

```bash
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc | cut -f1)
```

Thêm repository của Docker:

```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
```

Cài Docker và chạy thử:

```bash
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

Cho phép chạy Docker không cần `sudo` (lưu ý của Docker: thành viên nhóm `docker` có quyền
tương đương root trên máy đó):

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

(`groupadd` có thể báo nhóm đã tồn tại, bình thường.) Thoát hẳn phiên đăng nhập rồi vào lại;
nếu vẫn chưa nhận nhóm, khởi động lại máy. Sau đó kiểm tra:

```bash
docker run hello-world
docker --version
docker compose version
```

Đạt khi `docker run hello-world` (không `sudo`) in "Hello from Docker!".

Lưu ý: cổng mà container công bố ra ngoài (`-p`) bỏ qua luật `ufw`, nên `ufw` không thay thế
việc chỉ publish những cổng thật sự cần.

## 4. Lấy mã nguồn từ GitHub

### 4.1 Cấu hình Git
```bash
git config --global user.name "Ten Cua Ban"
git config --global user.email "email-github-cua-ban@example.com"
git config --global init.defaultBranch main
```

### 4.2 Tạo SSH key và thêm vào GitHub
Theo GitHub Docs, dùng Ed25519 (nên đặt passphrase; Enter để nhận vị trí file mặc định):

```bash
ssh-keygen -t ed25519 -C "email-github-cua-ban@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy toàn bộ dòng in ra (bắt đầu bằng `ssh-ed25519`). Trên GitHub: ảnh đại diện → Settings →
SSH and GPG keys → New SSH key → dán vào và đặt tiêu đề dễ nhận biết. Không bao giờ copy hay
gửi file private `id_ed25519` (file không có đuôi `.pub`).

### 4.3 Kiểm tra kết nối
```bash
ssh -T git@github.com
```

Lần đầu, SSH hiện fingerprint của GitHub và hỏi có tiếp tục không. Chỉ gõ `yes` nếu fingerprint
Ed25519 hiển thị đúng bằng giá trị GitHub công bố
(<https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints>):

```text
SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU
```

Đạt khi có thông báo dạng "Hi <username>! You've successfully authenticated, but GitHub does
not provide shell access."

### 4.4 Clone
```bash
cd ~
git clone git@github.com:XUANMAI1812/ecg-mlops.git
cd ecg-mlops
```

Từ đây trở đi, mọi lệnh chạy từ thư mục gốc repo (`ecg-mlops/`). Các đường dẫn tương đối và
`PYTHONPATH=src` bên dưới đều dựa vào điều này.

## 5. Tạo môi trường Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
```

Các phiên bản thư viện được ghim trong `requirements/dev.txt`. Mỗi lần mở terminal mới, kích
hoạt lại bằng `source .venv/bin/activate`. Nếu `pip install` báo `externally-managed-environment`
nghĩa là bạn chưa kích hoạt venv.

## 6. Chạy kiểm thử

Kiểm thử đơn vị dùng dữ liệu tổng hợp nhỏ, không cần tải PTB-XL:

```bash
PYTHONPATH=src pytest -q tests/unit
ruff check src tests
ruff format --check src tests
```

Đạt khi: `19 passed`, `All checks passed!`, và `ruff format --check` báo các file đã đúng
định dạng. Các test này dùng dữ liệu giả lập nên không thay thế bước khảo sát trên dữ liệu
thật ở mục 8.

## 7. Lấy dữ liệu PTB-XL v1.0.3 qua DVC

Nguồn gốc dữ liệu: <https://physionet.org/content/ptb-xl/1.0.3/> (giấy phép CC BY 4.0). Dữ liệu
không nằm trực tiếp trong git — được quản lý bằng DVC, lưu thật trên AWS S3; git chỉ theo dõi
file con trỏ nhỏ `data/raw/ptb-xl.dvc` (đã có sẵn khi `git clone` ở mục 4).

Xin access key (access key ID + secret access key) từ người quản lý AWS của nhóm, qua kênh an
toàn (chat riêng, không dán vào file repo hoặc gửi qua commit message). Cấu hình key — lệnh
`--local` ghi vào `.dvc/config.local`, file này không commit vào git:

```bash
dvc remote modify --local s3remote access_key_id '<ACCESS_KEY_ID_CỦA_BẠN>'
dvc remote modify --local s3remote secret_access_key '<SECRET_ACCESS_KEY_CỦA_BẠN>'
```

Kéo dữ liệu từ S3:

```bash
dvc pull
ls data/raw/ptb-xl
```

Đạt khi `ls` cho ra đủ: `records100/`, `records500/`, `ptbxl_database.csv`,
`scp_statements.csv`, `SHA256SUMS.txt`, `example_physionet.py`, `LICENSE.txt`, `RECORDS`,
hai file changelog. DVC tự kiểm tra tính toàn vẹn qua hash nội dung khi `pull`, không cần chạy
`sha256sum` thủ công.

Cấu hình đường dẫn dữ liệu:

```bash
cp .env.example .env
set -a; . ./.env; set +a
echo "$PTBXL_DIR"
ls "$PTBXL_DIR"
```

Lệnh `set -a; . ./.env; set +a` nạp các biến trong `.env` vào phiên terminal hiện tại. Cần chạy
lại mỗi khi mở terminal mới (hoặc truyền `--data-dir` thay thế). Không commit file `.env`.

## 8. Khảo sát dữ liệu

### 8.1 Đọc thử một bản ghi

Nếu đang ở phiên terminal mới thì venv chưa được kích hoạt lại — kích hoạt trước (thiếu bước
này, lệnh `python` báo `Command 'python' not found`, chỉ `python3` có sẵn trên hệ thống):

```bash
source .venv/bin/activate
PYTHONPATH=src python -m ecg.data.load --ecg-id 1
```

Script đọc bản ghi 100 Hz, rồi tự kiểm tra: shape phải là `(1000, 12)` (10 giây × 100 Hz,
12 chuyển đạo), `fs` trong header phải là 100, header có 12 tên chuyển đạo, không có NaN/inf.
Đạt thì in thông tin bản ghi (`shape`, `dtype`, `fs`, `leads`, `min / max`, `scp_codes`,
`strat_fold`) kèm dòng `validate     : OK ...` và thoát mã 0. Sai thì in `LỖI: ...` kèm lý do và
thoát mã 1. Lưu ý: script chưa kiểm tra tên và thứ tự chuyển đạo, chỉ kiểm tra số lượng.

Thử bản 500 Hz (kỳ vọng `shape` là `(5000, 12)`):

```bash
PYTHONPATH=src python -m ecg.data.load --ecg-id 1 --sampling-rate 500
```

`ecg_id` không liên tục (một số bản ghi bị gỡ ở các bản phát hành sau), nên nếu một `ecg_id`
báo không có trong `ptbxl_database.csv` thì thử id khác.

### 8.2 Thống kê nhãn và fold

```bash
PYTHONPATH=src python -m ecg.data.labels
```

Script in: số bản ghi, số bệnh nhân, số fold tối đa mà một bệnh nhân xuất hiện, số bản ghi
theo `strat_fold`, số bản ghi theo 5 superclass (bản ghi đa nhãn được đếm ở mọi lớp của nó nên
tổng có thể lớn hơn số bản ghi), số bản ghi không có superclass nào, số bản ghi có từ 2
superclass trở lên, và tỉ lệ `validated_by_human` theo fold.

Cách gộp nhãn bám theo `example_physionet.py` đi kèm dataset: chỉ giữ mã SCP có
`diagnostic == 1` trong `scp_statements.csv`, rồi tra cột `diagnostic_class`; không lọc theo
độ chắc chắn (likelihood) của từng mã.

### 8.3 Đối chiếu số liệu

Cột giữa là số ghi trên trang dataset v1.0.3, KHÔNG phải số đã đo. Số đo thật là số bạn chạy
ra ở mục 8.2; nếu lệch, ghi cả hai con số và báo nhóm, không tự sửa vào báo cáo.

| Mục | Trang PhysioNet v1.0.3 |
|---|---|
| Số bản ghi | 21799 (phần meta của trang ghi 21801) |
| Số bệnh nhân | 18869 |
| NORM | 9514 |
| MI | 5469 |
| STTC | 5235 |
| CD | 4898 |
| HYP | 2649 |

Trang dataset cũng ghi mọi bản ghi của một bệnh nhân nằm cùng một fold (nên "số fold tối đa
của một bệnh nhân" kỳ vọng là 1), và fold 9, 10 đã qua ít nhất một lần đánh giá của bác sĩ.

## 9. Xử lý sự cố nhanh

| Triệu chứng | Hướng xử lý |
|---|---|
| `Permission denied (publickey)` khi clone | Chưa thêm SSH key vào GitHub, chưa chấp nhận lời mời collaborator, hoặc chưa chạy `ssh-add`. Chạy lại mục 4.1, 4.3, 4.4 |
| `docker: permission denied` | Chưa vào lại phiên sau `usermod -aG docker`; thoát và đăng nhập lại (hoặc khởi động lại máy) |
| `externally-managed-environment` | Chưa kích hoạt venv: `source .venv/bin/activate` |
| `Command 'python' not found` | Chưa kích hoạt venv trong phiên này: `source .venv/bin/activate` (lệnh `python` chỉ có khi venv active, hệ thống chỉ có `python3`) |
| `ModuleNotFoundError: ecg` | Chạy từ thư mục gốc repo và thêm `PYTHONPATH=src` trước lệnh |
| `Không thấy .../ptbxl_database.csv` | `PTBXL_DIR` chưa được nạp hoặc sai. Chạy `echo "$PTBXL_DIR"`, hoặc truyền `--data-dir` |
| `dvc pull` báo `AccessDenied` | Access key chưa đúng hoặc chưa cấu hình — hỏi lại người quản lý AWS, chạy lại `dvc remote modify --local` ở mục 7 |
| `dvc pull` báo không thấy remote / lỗi đọc `.dvc/config` | Kiểm tra `dvc remote list`; đảm bảo đã `git clone`/`git pull` đầy đủ để có `.dvc/config` |
| `dvc: command not found` | Chưa kích hoạt venv (`source .venv/bin/activate`) hoặc `pip install -r requirements/dev.txt` chưa chạy xong |
| Đĩa đầy | `df -h /`, `docker system df`; mở rộng ổ đĩa |

## 10. Trích dẫn dữ liệu

- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a
  large publicly available electrocardiography dataset (version 1.0.3). PhysioNet.
  <https://doi.org/10.13026/kfzx-aw45>
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W.,
  Schaeffter, T. (2020). PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data.
  <https://doi.org/10.1038/s41597-020-0495-6>
