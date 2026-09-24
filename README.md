# 即時虛擬背景 — Real-time Virtual Background
## 使用 MODNet + OpenCV + PyTorch

---

## 專案結構

```
virtual_bg_project/
├── virtual_bg.py          ← 主程式（執行這個）
├── modnet_model.py        ← MODNet 模型架構定義
├── modnet_webcam_portrait_matting.ckpt  ← 預訓練權重（需自行下載）
├── bg_beach.jpg           ← 可選：自備背景圖 1
├── bg_city.jpg            ← 可選：自備背景圖 2
└── README.md
```

---

## 環境安裝

```bash
pip install torch torchvision opencv-python numpy Pillow
```

> GPU 加速（可選）：依照 https://pytorch.org 安裝對應 CUDA 版本的 torch

---

## 下載預訓練權重

1. 下載網址：https://huggingface.co/XM5354/Modnet_models/resolve/main/modnet_webcam_portrait_matting.ckpt"

2. 下載 `modnet_webcam_portrait_matting.ckpt`

3. 放到與 `virtual_bg.py` 相同的目錄

---

## 執行

```bash
python virtual_bg.py
```

---

## 操作說明

| 按鍵 | 功能 |
|------|------|
| `1` | 背景 1：海洋漸層 |
| `2` | 背景 2：日落漸層 |
| `3` | 背景 3：森林漸層 |
| `4` | 背景 4：Bokeh 模糊（或自備 bg_beach.jpg）|
| `5` | 背景 5：馬賽克（或自備 bg_city.jpg）|
| `B` | 切換 Mask 顯示（左半 mask / 右半合成，Debug 用）|
| `Q` / `ESC` | 離開程式 |

---

## 加入自訂背景圖

把任意圖片命名為 `bg_beach.jpg` 或 `bg_city.jpg` 放在同目錄即可，
程式啟動時會自動偵測並載入，替換掉預設的程式碼產生背景。

---

## 核心技術說明

### 1. 語義分割 — MODNet
- 輸入：攝影機 RGB 幀 (H × W × 3)
- 輸出：alpha matte (H × W)，每像素值 0~1 代表「是人的機率」
- 特點：輕量（MobileNetV2 backbone），可在 CPU 上達到 ~10 FPS

### 2. 前處理關鍵步驟
```python
# BGR → RGB（OpenCV 預設 BGR，模型訓練用 RGB）
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 確保長寬為 32 的倍數（Encoder 下採樣對齊需求）
rh = h - (h % 32)
rw = w - (w % 32)

# HWC → NCHW（PyTorch 格式）
tensor = torch.from_numpy(frame_resized).float() / 255.0
tensor = tensor.permute(2, 0, 1).unsqueeze(0)
```

### 3. Alpha Blending 合成公式
```
result = foreground × α + background × (1 - α)
```
- α = MODNet 輸出的 matte
- α 接近 1 → 顯示前景（人）
- α 接近 0 → 顯示背景
- α 介於中間 → 半透明過渡（頭髮、耳朵邊緣）

---

## 常見問題

**Q：程式跑得很慢（< 5 FPS）**
A：CPU 推論的正常現象。可以在 `virtual_bg.py` 中調低攝影機解析度：
```python
DISPLAY_WIDTH  = 640
DISPLAY_HEIGHT = 480
```

**Q：攝影機開不起來**
A：把 `CAMERA_INDEX = 0` 改成 `1` 或 `2` 試試。

**Q：找不到權重檔案**
A：確認 `.ckpt` 檔案名稱是否完全一致，並放在與 `virtual_bg.py` 相同目錄。