"""
virtual_bg.py — 即時虛擬背景 / 人像去背
使用 MODNet + OpenCV + Alpha Blending

操作說明：
  數字鍵 1~5  → 切換背景
  B           → 切換是否顯示黑白 Mask（Debug 用）
  Q / ESC     → 離開程式
"""

# ═══════════════════════════════════════════════════════════════════
#  匯入函式庫
#  這個區塊負責載入程式執行所需的所有外部模組。
#  分成四組：標準函式庫、影像處理、深度學習、模型定義。
# ═══════════════════════════════════════════════════════════════════

print("[1/6] 載入標準函式庫...", flush=True)
import sys   # 用於操作 Python 執行路徑（sys.path）與強制結束程式（sys.exit）
import os    # 用於檔案路徑操作（os.path.exists / os.path.abspath）
import time  # 用於計算 FPS（測量每幀處理時間）

print("[2/6] 載入 cv2 / numpy...", flush=True)
import cv2        # OpenCV：負責攝影機讀取、影像縮放、顯示視窗、鍵盤偵測
import numpy as np  # NumPy：負責影像的矩陣運算（像素合成、漸層產生等）

print("[3/6] 載入 PyTorch...", flush=True)
import torch                       # PyTorch 核心：張量運算、GPU 支援
import torch.nn as nn              # 神經網路模組（用於包裝 MODNet）
import torch.nn.functional as F    # 常用的神經網路函數（備用，模型內部會用到）

# ── MODNet 模型定義載入 ──────────────────────────────────────────
# 優先使用 MODNet/ 資料夾內的官方原始碼；
# 若官方資料夾不存在，才退而使用本地的 modnet_model.py 備用版本。
print("[4/6] 載入 MODNet 模型定義...", flush=True)
_base = os.path.dirname(os.path.abspath(__file__))          # 取得本程式所在的資料夾路徑
_official_src = os.path.join(_base, "MODNet", "src")        # 拼接官方 MODNet src 路徑
if os.path.exists(_official_src):                           # 若官方資料夾存在
    sys.path.insert(0, _official_src)                       # 將官方 src 插入 Python 搜尋路徑最前面
    from models.modnet import MODNet                        # 從官方路徑匯入 MODNet 類別
    print("       使用官方 MODNet src", flush=True)
else:                                                       # 若官方資料夾不存在（備用方案）
    sys.path.insert(0, _base)                               # 將本程式資料夾插入搜尋路徑
    from modnet_model import MODNet                         # 從本地備用檔案匯入 MODNet
    print("       使用本地 modnet_model.py (可能有 key mismatch)", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  設定區
#  集中管理所有可調整的參數，方便修改而不需要深入程式內部。
# ═══════════════════════════════════════════════════════════════════
WEIGHTS_PATH   = "modnet_webcam_portrait_matting.ckpt"  # 預訓練模型權重檔路徑（約 26 MB）
CAMERA_INDEX   = 0      # 攝影機編號：0 = 系統預設攝影機，多台時可改 1 / 2
DISPLAY_WIDTH  = 1280   # 要求攝影機輸出的寬度（像素）
DISPLAY_HEIGHT = 720    # 要求攝影機輸出的高度（像素）


# ═══════════════════════════════════════════════════════════════════
#  背景工具函式
#  這個區塊提供四種背景來源的產生方式：
#  1. 漸層純色（靜態，啟動時預先算好）
#  2. 高斯模糊（動態，每幀即時計算）
#  3. 馬賽克（動態，每幀即時計算）
#  4. 從檔案讀取圖片（靜態）
# ═══════════════════════════════════════════════════════════════════

def make_gradient_bg(h, w, color1, color2, vertical=True):
    """
    產生從 color1 到 color2 的線性漸層背景圖。
    color1 / color2 為 0~1 的 RGB 浮點數 tuple，例如 (0.05, 0.3, 0.8)。
    vertical=True 為垂直漸層（由上到下），False 為水平漸層（由左到右）。
    """
    bg = np.zeros((h, w, 3), dtype=np.float32)  # 建立一張全黑的空白影像，shape=(H, W, 3)，值域 0~1

    for i in range(h if vertical else w):  # 逐行（垂直）或逐列（水平）填色
        # t 是 0~1 之間的插值比例，代表目前掃到第幾成
        t = i / max(h - 1 if vertical else w - 1, 1)

        # 對 R、G、B 三個通道分別做線性插值：color = color1*(1-t) + color2*t
        color = tuple(c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2))

        if vertical:
            bg[i, :] = color   # 把整列（第 i 行的所有像素）設為同一顏色
        else:
            bg[:, i] = color   # 把整行（第 i 列的所有像素）設為同一顏色

    return (bg * 255).astype(np.uint8)  # 將 0~1 的浮點數轉為 0~255 的整數，供 OpenCV 使用


def make_blur_bg(frame):
    """
    對當前攝影機畫面做高斯模糊，產生景深虛化（Bokeh）效果的背景。
    kernel size (55, 55) 越大模糊程度越高；sigma=0 表示自動計算。
    """
    return cv2.GaussianBlur(frame, (55, 55), 0)  # 用 55x55 的高斯核心對整張影像做模糊


def make_mosaic_bg(frame, block=24):
    """
    對當前畫面做馬賽克處理：先縮小再放大，讓細節變成明顯色塊。
    block=24 表示每個馬賽克方塊為 24x24 像素。
    """
    # 第一步：將影像縮小到原本的 1/block 大小（用雙線性插值，避免縮小時出現鋸齒）
    small = cv2.resize(frame, (frame.shape[1] // block, frame.shape[0] // block),
                       interpolation=cv2.INTER_LINEAR)
    # 第二步：將縮小的影像放大回原始尺寸（用最近鄰插值，保留方塊感）
    return cv2.resize(small, (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def load_image_bg(path, h, w):
    """
    從磁碟讀取自訂背景圖片，並縮放到攝影機解析度 (w, h)。
    若檔案不存在則回傳 None（程式會自動改用動態背景替代）。
    """
    if not os.path.exists(path):  # 檔案不存在就直接回傳 None，不報錯
        return None
    img = cv2.imread(path)               # 讀取圖片（OpenCV 預設 BGR 色彩順序）
    return cv2.resize(img, (w, h))       # 縮放到與攝影機相同的解析度


# ═══════════════════════════════════════════════════════════════════
#  前處理 / 後處理 / 合成
#  這是整個系統的核心流程：
#    攝影機幀 → preprocess → MODNet 推論 → postprocess_matte → alpha_blend → 輸出
# ═══════════════════════════════════════════════════════════════════

def preprocess(frame_bgr):
    """
    將 OpenCV 讀到的原始攝影機幀轉換成 MODNet 模型需要的輸入格式。

    轉換步驟：
      1. BGR → RGB（OpenCV 預設 BGR，但模型訓練時用 RGB）
      2. 調整尺寸為 32 的倍數（MODNet encoder 連續下採樣 5 次，需要能被 32 整除）
      3. numpy HWC → PyTorch NCHW 張量，值域從 0~255 正規化到 0~1
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # 將顏色順序從 BGR 轉成 RGB

    h, w = frame_rgb.shape[:2]          # 取得原始影像的高度與寬度
    rh = max(h - (h % 32), 32)          # 高度對齊到 32 的倍數（去掉餘數）；最小為 32
    rw = max(w - (w % 32), 32)          # 寬度對齊到 32 的倍數；最小為 32

    frame_resized = cv2.resize(frame_rgb, (rw, rh))  # 縮放到對齊後的尺寸

    tensor = torch.from_numpy(frame_resized).float() / 255.0  # numpy → float32 張量，並正規化到 0~1
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)             # HWC → CHW，再加 batch 維度 → NCHW (1,3,H,W)

    return tensor, (h, w)  # 回傳張量 & 原始尺寸（後處理時需要還原）


def postprocess_matte(matte_tensor, original_hw):
    """
    將 MODNet 輸出的 alpha matte 張量轉換回可用於合成的 numpy 陣列。

    步驟：
      1. 去除多餘維度（squeeze），從 GPU 搬回 CPU（.cpu()），轉為 numpy
      2. 縮放回攝影機原始解析度（模型推論用的是縮小後的尺寸）
    """
    matte = matte_tensor.squeeze().cpu().numpy()             # 移除 batch/channel 維度，搬回 CPU，轉 numpy
    matte = cv2.resize(matte, (original_hw[1], original_hw[0]))  # 縮放回原始解析度 (w, h)
    return matte  # shape=(H, W)，值域 0~1，接近 1 代表「是人」


def alpha_blend(foreground, background, alpha):
    """
    用 alpha matte 將前景（人）與背景合成為最終畫面。

    合成公式：result = foreground × α + background × (1 − α)
      α 接近 1 → 顯示前景（人體）
      α 接近 0 → 顯示背景
      α 介於中間 → 半透明過渡（頭髮邊緣、耳朵旁等）

    Args:
      foreground: 攝影機原始幀，shape=(H, W, 3)，uint8
      background: 背景圖，shape=(H, W, 3)，uint8
      alpha:      MODNet 輸出的 matte，shape=(H, W)，float32，值域 0~1
    """
    # alpha 目前是單通道 (H, W)，需要複製成三通道 (H, W, 3) 才能和 BGR 影像做乘法
    alpha_3ch = np.stack([alpha] * 3, axis=2)

    # 前景乘以 α，背景乘以 (1-α)，相加得到合成結果
    result = foreground.astype(np.float32) * alpha_3ch + \
             background.astype(np.float32) * (1.0 - alpha_3ch)

    return result.astype(np.uint8)  # 轉回 uint8（0~255）供 OpenCV 顯示


# ═══════════════════════════════════════════════════════════════════
#  OSD（On-Screen Display）畫面資訊疊加
#  在畫面左上角繪製半透明黑底資訊欄，顯示目前背景名稱、FPS 等狀態。
# ═══════════════════════════════════════════════════════════════════

def draw_osd(frame, bg_name, fps, show_mask):
    """
    在畫面左上角疊加半透明資訊欄（背景名稱、FPS、Mask 狀態、操作提示）。
    使用 addWeighted 混合達到半透明效果，避免文字直接蓋掉畫面。
    """
    overlay = frame.copy()  # 複製一份畫面，用來畫黑底（避免直接修改原始幀）

    # 在複製幀上畫一個純黑矩形（左上角 (0,0) 到 (400,115)），作為文字底板
    cv2.rectangle(overlay, (0, 0), (400, 115), (0, 0, 0), -1)

    # 將黑底 overlay 以 45% 透明度疊加到原始 frame 上，產生半透明效果
    # addWeighted(src1, alpha, src2, beta, gamma, dst)
    # → frame = overlay*0.45 + frame*0.55 + 0
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 要顯示的四行文字內容
    lines = [
        f"BG: {bg_name}",                               # 目前背景名稱
        f"FPS: {fps:.1f}",                              # 目前處理速度（每秒幾幀）
        f"Mask: {'ON' if show_mask else 'OFF'}  (B toggle)",  # Mask 顯示狀態
        "1~5 change BG    Q/ESC quit",                  # 操作提示
    ]
    # 各行文字的顏色（BGR）：白、綠、黃、灰
    colors = [(255,255,255), (100,230,100), (200,200,100), (150,150,150)]

    # 逐行繪製文字，每行間距 24 像素
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line,
                    (12, 24 + i * 24),           # 文字起始位置（x=12, y 每行加 24）
                    cv2.FONT_HERSHEY_SIMPLEX,     # 字型：OpenCV 內建的標準字型
                    0.58,                         # 字體大小
                    color,                        # 文字顏色
                    1,                            # 線條粗細
                    cv2.LINE_AA)                  # 抗鋸齒，讓文字邊緣更平滑
    return frame


# ═══════════════════════════════════════════════════════════════════
#  主程式
#  main() 是整個應用程式的入口，負責依序完成：
#    1. 選擇計算裝置（GPU / CPU）
#    2. 驗證並載入模型權重
#    3. 開啟攝影機
#    4. 預先產生靜態背景
#    5. 進入主迴圈：每幀執行「讀幀 → 推論 → 合成 → 顯示」
# ═══════════════════════════════════════════════════════════════════

def main():

    # ── 1. 選擇計算裝置 ──────────────────────────────────────────
    # 偵測系統是否有可用的 NVIDIA GPU（CUDA）；有就用 GPU，沒有就退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[5/6] 執行裝置：{device}", flush=True)

    # ── 2. 驗證權重檔 ────────────────────────────────────────────
    if not os.path.exists(WEIGHTS_PATH):  # 確認 .ckpt 檔案存在
        print(f"\n[錯誤] 找不到權重檔：{WEIGHTS_PATH}")
        print(f"       請確認檔案放在：{os.path.abspath(WEIGHTS_PATH)}")
        sys.exit(1)  # 找不到就中止程式，避免後續報錯更難理解

    ckpt_size_mb = os.path.getsize(WEIGHTS_PATH) / 1024 / 1024  # 計算檔案大小（單位：MB）
    print(f"       權重檔大小：{ckpt_size_mb:.1f} MB", flush=True)
    if ckpt_size_mb < 20:  # 正常完整的權重約 26 MB，小於 20 MB 代表下載不完整
        print("  [警告] 檔案太小，可能下載不完整！正確大小約 26 MB")
        sys.exit(1)

    # ── 3. 建立模型並載入預訓練權重 ──────────────────────────────
    print("[5/6] 載入 MODNet 模型權重（約 5~20 秒，請稍候）...", flush=True)
    t0 = time.time()  # 記錄開始時間，用於計算載入耗時

    modnet = MODNet(backbone_pretrained=False)  # 建立 MODNet 模型實例（不下載 backbone 預訓練權重）
    modnet = nn.DataParallel(modnet)            # 用 DataParallel 包裝，支援多 GPU；單 GPU 也適用

    try:
        state = torch.load(WEIGHTS_PATH, map_location=device)  # 讀取 .ckpt 權重，直接載到目標裝置
        modnet.load_state_dict(state)                          # 將權重填入模型的每一層
    except Exception as e:
        print(f"\n[錯誤] 模型權重載入失敗：{e}")
        print("       最可能原因：ckpt 檔案下載不完整或格式錯誤")
        sys.exit(1)

    modnet.eval()       # 切換到推論模式：關閉 Dropout / BatchNorm 的訓練行為
    modnet.to(device)   # 將整個模型移到 GPU 或 CPU
    print(f"       模型載入完成，耗時 {time.time()-t0:.1f} 秒", flush=True)

    # ── 4. 開啟攝影機 ────────────────────────────────────────────
    print(f"[6/6] 開啟攝影機（index={CAMERA_INDEX}）...", flush=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)  # 建立攝影機物件，CAMERA_INDEX=0 代表系統預設攝影機
    if not cap.isOpened():                # 若無法開啟攝影機（未連接或被佔用）
        print(f"[錯誤] 無法開啟攝影機（index={CAMERA_INDEX}）")
        print("       請確認攝影機已連接，或把 CAMERA_INDEX 改成 1 / 2")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)   # 要求攝影機輸出 1280 像素寬
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)  # 要求攝影機輸出 720 像素高

    ret, first_frame = cap.read()  # 嘗試讀取第一幀，確認攝影機真的有畫面輸出
    if not ret:
        print("[錯誤] 攝影機開啟了但無法讀取畫面")
        sys.exit(1)

    H, W = first_frame.shape[:2]  # 取得實際解析度（攝影機不一定接受要求的尺寸）
    print(f"       攝影機解析度：{W}x{H}", flush=True)

    # ── 5. 預先產生靜態背景圖 ────────────────────────────────────
    # 靜態背景在啟動時算好存起來，不需要每幀重新計算，節省效能
    print("       準備背景圖...", flush=True)
    beach_bg = load_image_bg("bg_beach.jpg", H, W)  # 嘗試讀取海灘圖片（若不存在回傳 None）
    city_bg  = load_image_bg("bg_city.jpg",  H, W)  # 嘗試讀取城市圖片（若不存在回傳 None）

    # 定義 5 個背景選項的字典清單，type 決定每幀如何產生背景：
    #   "static"        → 使用預先算好的圖片（img 欄位）
    #   "dynamic"       → 每幀即時高斯模糊
    #   "dynamic_mosaic"→ 每幀即時馬賽克
    BACKGROUNDS = [
        {"name": "海洋漸層",  "type": "static",
         "img": make_gradient_bg(H, W, (0.05,0.3,0.8), (0.4,0.85,0.95))},   # 深藍→淺藍漸層

        {"name": "日落漸層",  "type": "static",
         "img": make_gradient_bg(H, W, (0.1,0.05,0.4), (0.95,0.45,0.1))},   # 深紫→橙紅漸層

        {"name": "森林漸層",  "type": "static",
         "img": make_gradient_bg(H, W, (0.05,0.35,0.1),(0.6,0.9,0.4))},     # 深綠→亮綠漸層

        # 若 bg_beach.jpg 存在就用圖片，否則改用即時景深模糊
        {"name": "景深模糊 (Bokeh)" if beach_bg is None else "海灘圖片",
         "type": "dynamic" if beach_bg is None else "static",
         "img": beach_bg},

        # 若 bg_city.jpg 存在就用圖片，否則改用即時馬賽克
        {"name": "馬賽克" if city_bg is None else "城市圖片",
         "type": "dynamic_mosaic" if city_bg is None else "static",
         "img": city_bg},
    ]

    print("\n===== 啟動成功！視窗應在 3 秒內出現 =====", flush=True)
    print("操作：數字 1~5 切換背景 | B 顯示/隱藏 Mask | Q 或 ESC 離開\n", flush=True)

    # ── 6. 初始化主迴圈變數 ──────────────────────────────────────
    current_bg_idx = 0       # 目前選擇的背景索引（0~4），初始為第 1 個（海洋漸層）
    show_mask = False        # 是否開啟 Mask Debug 模式（左半顯示灰階 matte）
    prev_time = time.time()  # 上一幀的時間戳，用於計算 FPS
    frame_count = 0          # 幀計數器，用於第一幀的啟動確認訊息

    # ═══════════════════════════════════════════════════════════════
    #  主迴圈
    #  每次迭代處理一幀影像，流程：
    #    讀幀 → 鍵盤 → 前處理 → 推論 → 後處理 → 選背景 → 合成 → 顯示
    # ═══════════════════════════════════════════════════════════════
    while True:

        # ── 讀取攝影機幀 ──────────────────────────────────────────
        ret, frame = cap.read()   # ret=True 代表讀取成功；frame 是 BGR numpy 陣列 (H, W, 3)
        if not ret:
            print("[警告] 丟幀，跳過", flush=True)  # 攝影機偶爾掉幀屬正常，跳過繼續
            continue

        frame_count += 1  # 每成功讀到一幀就累加計數

        if frame_count == 1:
            print("[OK] 收到第一幀，開始推論...", flush=True)  # 第一幀啟動確認

        # ── 鍵盤事件偵測 ──────────────────────────────────────────
        # waitKey(1) 等待 1 毫秒鍵盤輸入，同時讓 OpenCV 視窗保持響應
        # & 0xFF 是遮罩操作，取低 8 位元，確保跨平台的按鍵值一致
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):     # Q 鍵 或 ESC（ASCII 27）→ 離開主迴圈
            break
        elif key in (ord('b'), ord('B')):        # B 鍵 → 切換 Mask 顯示模式
            show_mask = not show_mask
        elif ord('1') <= key <= ord('5'):        # 數字鍵 1~5 → 切換背景
            current_bg_idx = key - ord('1')      # 將字元 '1'~'5' 轉為索引 0~4

        # ── MODNet 推論（核心步驟）──────────────────────────────
        input_tensor, original_hw = preprocess(frame)   # 將攝影機幀轉為模型輸入張量
        input_tensor = input_tensor.to(device)          # 將輸入資料移到 GPU（或 CPU）

        with torch.no_grad():  # 推論時不需要計算梯度，可節省約 50% 記憶體並加快速度
            # MODNet 回傳三個值：semantic（語意分割）、detail（細節）、matte（最終 alpha）
            # 我們只需要第三個 matte 輸出，前兩個用 _ 忽略
            _, _, matte_tensor = modnet(input_tensor, inference=True)

        matte = postprocess_matte(matte_tensor, original_hw)  # 將 matte 轉回 numpy，縮放至原始尺寸

        if frame_count == 1:
            print("[OK] 第一幀推論完成，視窗應已出現！", flush=True)

        # ── 選擇並準備背景 ────────────────────────────────────────
        bg_info = BACKGROUNDS[current_bg_idx]  # 取得目前選擇的背景資訊字典

        if bg_info["type"] == "static":
            # 靜態背景：直接用預先算好的圖片；若圖片為 None（讀取失敗）就用灰色漸層替代
            background = bg_info["img"] if bg_info["img"] is not None else \
                         make_gradient_bg(H, W, (0.2,0.2,0.2), (0.5,0.5,0.5))
        elif bg_info["type"] == "dynamic":
            background = make_blur_bg(frame)    # 動態景深：對當前幀做高斯模糊
        else:
            background = make_mosaic_bg(frame)  # 動態馬賽克：對當前幀做馬賽克處理

        # ── Alpha Blending 合成 ───────────────────────────────────
        # 用 MODNet 輸出的 matte 將人像前景與虛擬背景疊合
        result = alpha_blend(frame, background, matte)

        # ── Mask Debug 模式 ───────────────────────────────────────
        if show_mask:
            # 將 matte 值域 0~1 轉為 0~255 的灰階影像，再轉成三通道方便疊合
            mask_vis = cv2.cvtColor((matte*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            half = W // 2                                    # 畫面水平中線
            result[:, :half] = mask_vis[:, :half]            # 左半邊顯示灰階 matte
            cv2.line(result, (half,0),(half,H),(100,230,100),1)  # 畫一條綠線分隔左右

            # 在左下角標示 "MASK"，右下角標示 "OUTPUT"
            cv2.putText(result,"MASK",   (10,    H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,230,100), 1, cv2.LINE_AA)
            cv2.putText(result,"OUTPUT", (half+10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,230,100), 1, cv2.LINE_AA)

        # ── FPS 計算與 OSD 疊加 ───────────────────────────────────
        now = time.time()                           # 取得當前時間
        fps = 1.0 / max(now - prev_time, 1e-9)     # FPS = 1 / 兩幀間的時間差；1e-9 防止除以零
        prev_time = now                             # 更新上一幀時間戳

        result = draw_osd(result, bg_info["name"], fps, show_mask)  # 在畫面上疊加資訊欄

        # ── 顯示結果 ──────────────────────────────────────────────
        cv2.imshow("Virtual Background - MODNet + OpenCV", result)  # 在視窗中顯示合成後的畫面

    # ── 結束清理 ──────────────────────────────────────────────────
    cap.release()           # 釋放攝影機資源（讓其他程式可以再次使用攝影機）
    cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
    print("\n[結束] 程式正常離開。")


# ── 程式進入點 ────────────────────────────────────────────────────
# 只有直接執行本檔案時才呼叫 main()；
# 若被其他程式 import 則不執行（__name__ 此時不等於 "__main__"）
if __name__ == "__main__":
    main()
