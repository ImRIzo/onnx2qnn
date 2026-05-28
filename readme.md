# onnx2qnn

Convert YOLOv8 ONNX → int8 QNN context binary for **Qualcomm HTP NPUs**.

Takes your trained YOLOv8 `.onnx` file and produces a `.bin` that runs on the NPU.
Defaults target the **Radxa Dragon Q6A** (QCS6490 / Hexagon V68 / soc_id 35).

---

## Project structure

```
onnx2qnn/
│
├── best.onnx                        # Your YOLOv8 ONNX model (input)
├── val2017/                         # Calibration images (~100 JPEGs), I used COCO2017 validation images
│
├── onnx2qnn.sh                      # Main script — runs the whole pipeline
├── prepare_onnx.py                  # Splits YOLOv8 output into boxes + scores
├── prepare_calib.py                 # Generates raw calibration data from images
├── generate_binary_ondevice.sh      # Helper: generate .bin on the ARM device
│
├── config_file.json                 # Backend config wrapper (edit for your SoC)
├── htp_backend_extensions.json      # Backend config (soc_id, dsp_arch, vtcm_mb)
│
├── .gitignore
└── readme.md                        # This file
```

After running the pipeline:
```
export/
├── yolov8_q6a.bin                   # The context binary (~4 MB)
├── yolov8_q6a_config.json           # Runtime config (scales, tensor names)
└── work/                            # Intermediate files (safe to delete)
```

---

## Requirements

| What | How |
|---|---|
| Linux x86-64 | Build machine. Tested on Ubuntu 22.04 with Python 3.10. |
| QAIRT SDK 2.45.40 | [Download from Qualcomm](https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.40.260406/v2.45.40.260406.zip), then `source /path/to/qairt/bin/envsetup.sh` |
| Python 3.10+ | `pip install onnx opencv-python numpy` |
| Calibration images | COCO val2017 or any ~100 JPEGs in `val2017/` |

---
** sometimes downloading sdk with web browser fails. better use wget **
## SoC configuration

Two JSON files in the project root control which hardware the binary targets.
They ship with QCS6490 defaults. Edit them for your SoC:

**`htp_backend_extensions.json`** — the actual params:
```json
{
    "graphs": [{ "graph_names": ["yolov8_det"], "vtcm_mb": 2 }],
    "devices": [{ "dsp_arch": "v68", "soc_id": 35 }]
}
```

**`config_file.json`** — wrapper that points to the backend config:
```json
{
    "backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so",
        "config_file_path": "htp_backend_extensions.json"
    }
}
```

| Parameter | QCS6490 | Meaning |
|---|---|---|
| `soc_id` | 35 | QCS6490 / SM7325 platform |
| `dsp_arch` | `"v68"` | Hexagon V68 |
| `vtcm_mb` | 2 | Vector Tightly Coupled Memory |

The pipeline reads these files at Step 5. No CLI flags needed.

---

## Usage

```bash
# 1. Activate the SDK
source /path/to/qairt/bin/envsetup.sh

# 2. Edit the configs for your hardware (or leave the QCS6490 defaults)
vim htp_backend_extensions.json

# 3. Run the pipeline:
./onnx2qnn.sh \
    --onnx best.onnx \
    --calib ./val2017/ \
    --prepare-all
```

Binary at `export/yolov8_q6a.bin`.

### Other workflows

```bash
# Already have a split-output ONNX, just need calibration:
./onnx2qnn.sh --ready-onnx best_ready.onnx --calib ./val2017/ --prepare-calib

# Everything already prepared — just convert:
./onnx2qnn.sh --ready-onnx best_ready.onnx --input-list ./calib_raw/input_list.txt
```

---

## What the pipeline does

### Step 1 — Split the ONNX output

YOLOv8 exports a single output `[1,84,8400]` — 4 box channels plus 80 class channels
glued together.

**Problem:** int8 quantization gives this one tensor a single scale. Since boxes
range 0–640 and scores range 0–1, the scale ends up ~2.5. Scores collapse to
0 or 1 — random garbage detections.

**Fix:** `prepare_onnx.py` splits it into two separate outputs, each with its own
quantization scale:

| Output | Shape | Scale |
|---|---|---|
| `/model.22/Mul_2_output_0` | `[1, 4, 8400]` | ~2.55 |
| `/model.22/Sigmoid_output_0` | `[1, 80, 8400]` | ~0.0038 |

### Step 2 — Calibration data

`prepare_calib.py` processes each image: resize to 640×640, BGR→RGB, normalize to
[0,1], transpose to CHW, add batch dimension. Saves as raw float32 files and
creates `input_list.txt`.

### Step 3 — Convert ONNX to DLC

`qairt-converter` converts the split ONNX into an unquantized Deep Learning
Container (`.dlc`) targeting the HTP backend.

### Step 4 — Quantize to int8

`qairt-quantizer` runs the model on CPU using the calibration data, collects
activation statistics, and assigns int8 encodings to every tensor.

### Step 5 — Generate context binary

`qnn-context-binary-generator` compiles the quantized DLC into a `.bin` that the
V68 HTP can execute directly. It packages actual V68 DSP skeleton libraries.

Reads SoC settings from `config_file.json` + `htp_backend_extensions.json` in
the project root.

### Step 6 — Config JSON

Extracts input/output tensor names, shapes, and quantization scales from the
quantized DLC. The runtime uses these to dequantize int8 outputs back to float.

---


## Troubleshooting

| Problem | Fix |
|---|---|
| SDK not found | `source /path/to/qairt/bin/envsetup.sh` first |
| Missing Python packages | `pip install onnx opencv-python numpy` | 
| Wrong detections on device | ONNX probably wasn't split. Run with `--prepare-onnx`. |
| VTCM size error on device | Set `vtcm_mb: 2` in `htp_backend_extensions.json` and rebuild. |
| Segfault on device | Use `--retrieve_context` with the .bin, not `--dlc_path`. |
| Zero quantization scales | Calibration files wrong size. Each `.raw` must be 4,915,200 bytes. |
| `Unused Input nodes found` | Safe to ignore. Optimizer pruning unused graph inputs. |

---

## References

- [Running a custom network on the Radxa Q6A Dragon](https://olof-astrand.medium.com/running-a-custom-network-on-the-radxa-q6a-dragon-bc4234db9eb3) — Olof Åstrand
- [Radxa Dragon Q6A NPU Docs](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage)
- QCS6490: `soc_id 35`, `dsp_arch v68`, SC7280 "Kodiak" silicon family
