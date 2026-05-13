# Realtime Object Tracker & Adaptive Input System

産業・軍事グレードのリアルタイム・オブジェクトトラッキング＆適応型入力デバイス変換システム。

## アーキテクチャ概要

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  DXGI Capture│────▶│  ONNX Runtime   │────▶│  Kalman Filter   │
│  (144+ fps)  │     │  (CUDA/TRT EP)  │     │  (6-state CA)    │
│  GPU→Staging │     │  YOLOv8 推論    │     │  未来位置予測    │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Input Control   │
                                              │  XInput→ViGEm    │
                                              │  EMA + 非線形減衰│
                                              └──────────────────┘
```

### モジュール構成

| モジュール | 技術 | 役割 |
|-----------|------|------|
| Capture   | DXGI Desktop Duplication API | 画面中央400x400をGPUから直接取得 (144fps+) |
| Inference | ONNX Runtime + CUDA EP | YOLOv8によるターゲット検出 |
| Tracking  | Kalman Filter (6-state) | 定加速度モデルによる未来位置予測 |
| Input     | XInput + ViGEmClient | 物理パッドのパススルー + AI右スティック補正 |

### スレッド間通信

ロックフリー SPSC リングバッファ（キャッシュライン整列、power-of-2 容量）によるゼロコピーに近いデータ受け渡し。

## 必要環境

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GeForce RTX 3060 以上（CUDA 11.x+）
- **コンパイラ**: MSVC 2019+ または MinGW-w64
- **CMake**: 3.20+

## 依存ライブラリ

| ライブラリ | 入手先 |
|-----------|--------|
| ONNX Runtime GPU | FetchContentで自動取得 (手動: https://github.com/microsoft/onnxruntime/releases) |
| OpenCV 4.x | FetchContentで自動取得 (手動: https://opencv.org/releases/) |
| ViGEmClient | FetchContentで自動取得 |
| ViGEmBus Driver | https://github.com/ViGEm/ViGEmBus/releases (要インストール) |
| CUDA Toolkit | https://developer.nvidia.com/cuda-downloads |
| YOLOv8 ONNX | Ultralytics等でエクスポート |

## ビルド手順

全依存ライブラリ (ONNX Runtime, OpenCV, ViGEmClient) は自動ダウンロードされます。
初回は ONNX Runtime SDK (~300MB) と OpenCV のダウンロード+ビルドで数分かかります。

### MSVC (Visual Studio)
```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
cmake --build build --config Release
```

### MinGW (CLion / コマンドライン)
```bash
# MinGW Makefiles を使用（Ninja ではなく）
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build

# または CMakePresets を使用
cmake --preset mingw-release
cmake --build --preset mingw-release-build
```

**CLion で使う場合:**
1. CLion でプロジェクトフォルダを開く
2. `Settings > Build > CMake` で Generator を **"MinGW Makefiles"** に変更
3. または `CMakePresets.json` が自動検出されるので、プリセットを選択

### 共通
```powershell
# YOLOv8 ONNXモデルを実行ファイルと同じディレクトリに配置
copy yolov8n.onnx build\RealtimeObjectTracker.exe のあるディレクトリ

# 実行
RealtimeObjectTracker.exe [model_path]
```

### オプション: ローカルSDKパス指定（高速化）
```
-DONNXRUNTIME_ROOT="C:/onnxruntime-win-x64-gpu-1.22.0"
-DOpenCV_DIR="C:/opencv/build"
```

## 補正ロジック詳細

### 非線形減衰カーブ
```
strength = pow(normalized_distance, decay_exponent)
```
- `normalized_distance` = ターゲットまでの距離 / 最大距離
- `decay_exponent` = 2.0（デフォルト）
- 目標に近づくほど出力が指数関数的に減衰 → オーバーシュート防止

### 指数平滑化 (EMA)
```
ema_output = α × raw_input + (1 - α) × ema_previous
```
- `α` = 0.35（デフォルト）
- 急激な入力変化を平滑化し、滑らかな吸い付きを実現

## 設定パラメータ

`SystemConfig` 構造体で全パラメータを一元管理：

```cpp
SystemConfig cfg;
cfg.smoothing_alpha      = 0.35f;   // EMA係数
cfg.decay_exponent       = 2.0f;    // 非線形減衰パワー
cfg.max_correction_speed = 20000.f; // 最大補正速度
cfg.confidence_threshold = 0.45f;   // 検出信頼度閾値
cfg.target_class_id      = 0;       // ターゲットクラスID
cfg.max_lost_frames      = 15;      // 検出喪失許容フレーム数
cfg.prediction_steps     = 3;       // 予測先読みステップ数
```

## ライセンス

MIT License
