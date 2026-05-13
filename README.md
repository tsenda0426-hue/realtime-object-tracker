# RT-DETR Realtime Adaptive Input Correction System

超低遅延・リアルタイム適応型入力補正システム。  
AIモデルに **RT-DETR** (Transformer ベース、NMS 不要) を採用した単一ファイルアーキテクチャ。

## アーキテクチャ概要

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  DXGI Capture│────▶│  ONNX Runtime   │────▶│  Kalman Filter   │
│  (144+ fps)  │     │  (CUDA EP)      │     │  (6-state CA)    │
│  GPU→Staging │     │  RT-DETR 推論   │     │  未来位置予測    │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  I/O Loop 1kHz   │
                                              │  XInput→ViGEm    │
                                              │  EMA + pow() 減衰│
                                              └──────────────────┘
```

### 4スレッド並列パイプライン

| スレッド | モジュール | 技術 | 役割 |
|----------|-----------|------|------|
| T1 | Capture   | DXGI Desktop Duplication API | 画面中央 400×400 を GPU から直接取得 |
| T2 | Inference | ONNX Runtime + CUDA EP       | RT-DETR によるターゲット検出 (NMS 不要) |
| T3 | Tracking  | Kalman Filter (6-state CA)   | 定加速度モデルで未来位置予測 |
| T4 | I/O       | XInput + ViGEmClient         | 1000 Hz で物理パッド→仮想パッド + AI 右スティック補正 |

### スレッド間通信

ロックフリー SPSC リングバッファ（キャッシュライン整列、power-of-2 容量）。

## 必要環境

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GeForce RTX 3060 以上（CUDA 11.x+）
- **コンパイラ**: MSVC 2019+ (C++20) または MinGW-w64
- **CMake**: 3.20+

## 依存ライブラリ

| ライブラリ | 入手先 |
|-----------|--------|
| ONNX Runtime GPU | FetchContent で自動取得 |
| OpenCV 4.x       | FetchContent で自動取得 |
| ViGEmClient      | FetchContent で自動取得 |
| ViGEmBus Driver  | https://github.com/ViGEm/ViGEmBus/releases (要インストール) |
| CUDA Toolkit     | https://developer.nvidia.com/cuda-downloads |
| RT-DETR ONNX     | Ultralytics / PaddleDetection でエクスポート |

## ビルド手順

全依存ライブラリは CMake FetchContent で自動ダウンロードされます。  
初回は ONNX Runtime SDK (~300MB) と OpenCV のダウンロード + ビルドで数分かかります。

### MSVC (Visual Studio)
```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
cmake --build build --config Release
```

### MinGW (CLion / コマンドライン)
```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build
```

### 実行
```powershell
# RT-DETR ONNX モデルを実行ファイルと同じディレクトリに配置
copy rtdetr.onnx build\

# 実行 (デフォルトモデル名: rtdetr.onnx)
RT_DETR_Tracker.exe [model_path]
```

### オプション: ローカル SDK パス指定
```
-DONNXRUNTIME_ROOT="C:/onnxruntime-win-x64-gpu-1.22.0"
-DOpenCV_DIR="C:/opencv/build"
```

## 補正ロジック

### 非線形減衰カーブ
```
strength = pow(normalized_distance, decay_exponent)
```
- `normalized_distance` = ターゲット距離 / 最大距離
- `decay_exponent` = 2.0（デフォルト）
- 目標に近づくほど出力が指数関数的に減衰 → オーバーシュート防止

### 指数平滑化 (EMA)
```
ema_output = α × raw + (1 - α) × ema_previous
```
- `α` = 0.35（デフォルト、速度に応じて 0.35〜0.95 に適応変化）

## プロジェクト構造

```
.
├── CMakeLists.txt      # C++20, FetchContent, MSVC/MinGW 最適化
├── src/
│   └── main.cpp        # 全モジュール統合 (単一ファイル)
└── README.md
```

## RT-DETR ONNX エクスポートの注意点

### Ultralytics でのエクスポート
```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")  # or rtdetr-x.pt
model.export(
    format="onnx",
    imgsz=640,
    opset=16,          # ONNX opset 16 推奨
    simplify=True,     # onnx-simplifier で最適化
    dynamic=False,     # 固定バッチサイズ=1 を推奨 (CUDA EP 最適化)
    half=False,        # FP16 は TensorRT EP 使用時のみ推奨
)
```

### 注意事項
1. **`dynamic=False`** を推奨。固定形状にすることで CUDA EP がグラフ最適化を最大限に適用でき、「Fallback mode」警告を防止します。
2. **`opset=16`** 以上を指定。RT-DETR の Multi-Head Attention 演算子は opset 16 で最も効率的に変換されます。
3. **`simplify=True`** を推奨。`onnx-simplifier` が冗長ノードを除去し、推論速度が向上します。
4. 本システムは以下の3つの出力フォーマットを自動判別します:
   - **Format A** (Ultralytics): `[1, 300, 6]` = `[x1, y1, x2, y2, score, class_id]`
   - **Format B** (PaddleDetection): `[1, 300, 6]` = `[class_id, score, x1, y1, x2, y2]`
   - **Format C** (Raw logits): `[1, 300, 4+C]` = `[cx, cy, w, h, cls0…clsC-1]`
5. NMS は RT-DETR の Transformer デコーダ内で暗黙的に処理されるため、外部 NMS は不要です。
