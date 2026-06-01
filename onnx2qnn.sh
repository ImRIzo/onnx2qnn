#!/bin/bash
# ============================================================================
#  YOLOv8 ONNX → QNN Context Binary (.bin) for Qualcomm HTP NPUs
#
#  Converts a YOLOv8 ONNX model into an int8-quantized context binary.
#  Targets Radxa Dragon Q6A by default (QCS6490 / V68 HTP / soc_id=35).
#
#  The ONNX is first split into 2 outputs (boxes + scores) for int8 quantization.
#  YOLOv8 design: boxes + scores.
#
#  Prerequisites:
#    1. QAIRT SDK installed and sourced: source /path/to/qairt/bin/envsetup.sh
#    2. Python 3 with: numpy, opencv-python, onnx
#    3. Calibration images (e.g. COCO val2017)
#    4. Edit config_file.json + htp_backend_extensions.json for your SoC
#
#  Usage:
#    ./onnx2qnn.sh --onnx best.onnx --calib ./val2017/ --prepare-all
#    ./onnx2qnn.sh --ready-onnx best_ready.onnx --calib ./val2017/ --prepare-calib
#    ./onnx2qnn.sh --ready-onnx best_ready.onnx --input-list ./calib_raw/input_list.txt
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
CALIB_NUM=100
OUTPUT_DIR="./export"
GRAPH_NAME="yolov8_det"
NUM_CLASSES=""                          # auto-detect if empty, else 1-80
INPUT_SHAPE="1,3,640,640"               # default static input B,C,H,W
DO_ONNX_SURGERY=0
DO_CALIB_GEN=0
SKIP_BINARY_GEN=0
INPUT_ONNX=""
READY_ONNX=""
CALIB_DIR=""
INPUT_LIST=""

# ═══════════════════════════════════════════════════════════════════════════════
#  Help
# ═══════════════════════════════════════════════════════════════════════════════
usage() {
    cat << 'HELPEOF'

Usage: onnx2qnn.sh [OPTIONS]

Converts YOLOv8 ONNX → int8-quantized QNN context binary (.bin).
SoC settings are read from config_file.json + htp_backend_extensions.json
in the project root. Edit those files for your hardware before running.

  INPUT (choose one):
    --onnx <file>              Raw YOLOv8 ONNX with single [1,84,8400] output.
                               Use with --prepare-onnx to split into boxes + scores.
    --ready-onnx <file>        Already-split ONNX (boxes [1,4,8400] + scores [1,80,8400]).

  PREPARATION:
    --prepare-onnx             Split [1,84,8400] → boxes + scores.
    --prepare-calib            Generate raw float32 calibration files from images.
    --prepare-all              -prepare-onnx + --prepare-calib.

  CALIBRATION (choose one):
    --calib <dir>              Directory of calibration images (*.jpg).
    --input-list <file>        Already-prepared input_list.txt.

  OPTIONS:
    --calib-num <n>            Number of calibration images (default: 100).
    --graph-name <name>        Graph name in DLC (default: yolov8_det).
    --output-dir <dir>         Output directory (default: ./export).
    --skip-binary-gen          Skip binary gen. Generate on-device later.
    --num-classes <n>          Number of classes (1-80). Auto-detect if omitted.
	    --input-shape <B,C,H,W>    Static input shape (default: 1,3,640,640). Removes dynamic dims for faster inference.
    --help, -h                 Show this message.

  QUICK START:
    ./onnx2qnn.sh --onnx best.onnx --calib ./val2017/ --prepare-all

HELPEOF
    exit 0
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Parse arguments
# ═══════════════════════════════════════════════════════════════════════════════
while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)           INPUT_ONNX="$(realpath "$2")";  shift 2 ;;
        --ready-onnx)     READY_ONNX="$(realpath "$2")";  shift 2 ;;
        --calib)          CALIB_DIR="$(realpath "$2")";   shift 2 ;;
        --input-list)     INPUT_LIST="$(realpath "$2")";  shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";                shift 2 ;;
        --calib-num)      CALIB_NUM="$2";                 shift 2 ;;
        --graph-name)     GRAPH_NAME="$2";                shift 2 ;;
        --prepare-onnx)   DO_ONNX_SURGERY=1;              shift   ;;
        --prepare-calib)  DO_CALIB_GEN=1;                 shift   ;;
        --prepare-all)    DO_ONNX_SURGERY=1; DO_CALIB_GEN=1; shift ;;
        --skip-binary-gen) SKIP_BINARY_GEN=1;              shift   ;;
        --num-classes)    NUM_CLASSES="$2";               shift 2 ;;
        --input-shape)    INPUT_SHAPE="$2";               shift 2 ;;
        --help|-h)        usage ;;
        *) echo "ERROR: Unknown option: $1"; usage ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
#  Validate inputs
# ═══════════════════════════════════════════════════════════════════════════════
FINAL_READY_ONNX=""
if [ -n "$READY_ONNX" ]; then
    FINAL_READY_ONNX="$READY_ONNX"
elif [ "$DO_ONNX_SURGERY" -eq 1 ] && [ -n "$INPUT_ONNX" ]; then
    : # produced in step 1
elif [ -n "$INPUT_ONNX" ] && [ "$DO_ONNX_SURGERY" -eq 0 ]; then
    FINAL_READY_ONNX="$INPUT_ONNX"
else
    echo "ERROR: No ONNX input. Use --ready-onnx <file> OR --onnx <file> --prepare-onnx"
    exit 1
fi

FINAL_INPUT_LIST=""
if [ -n "$INPUT_LIST" ]; then
    FINAL_INPUT_LIST="$INPUT_LIST"
elif [ "$DO_CALIB_GEN" -eq 1 ] && [ -n "$CALIB_DIR" ]; then
    : # produced in step 2
else
    echo "ERROR: No calibration data. Use --input-list <file> OR --calib <dir> --prepare-calib"
    exit 1
fi

# ── SDK check ────────────────────────────────────────────────────────────────
if [ -z "${QAIRT_SDK_ROOT:-}" ] && [ -z "${QNN_SDK_ROOT:-}" ]; then
    echo "ERROR: QAIRT SDK not found. Run: source /path/to/qairt/bin/envsetup.sh"
    exit 1
fi
SDK_ROOT="${QAIRT_SDK_ROOT:-$QNN_SDK_ROOT}"

QAIRT_CONVERTER="$SDK_ROOT/bin/x86_64-linux-clang/qairt-converter"
QAIRT_QUANTIZER="$SDK_ROOT/bin/x86_64-linux-clang/qairt-quantizer"
BIN_GEN="$SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
HTP_BACKEND="$SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"
DLC_INFO="$SDK_ROOT/bin/x86_64-linux-clang/qairt-dlc-info"
DLC_TO_JSON="$SDK_ROOT/bin/x86_64-linux-clang/qairt-dlc-to-json"

for tool in "$QAIRT_CONVERTER" "$QAIRT_QUANTIZER" "$BIN_GEN" "$HTP_BACKEND" "$DLC_INFO"; do
    if [ ! -f "$tool" ]; then
        echo "ERROR: Missing SDK tool: $tool"
        exit 1
    fi
done

export LD_LIBRARY_PATH="$SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="python3"

# ── Setup output dirs ────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
WORK_DIR="$OUTPUT_DIR/work"
mkdir -p "$WORK_DIR"

echo "============================================================"
echo "  YOLOv8 → QNN Context Binary Pipeline"
echo "============================================================"
echo "  SDK:        $SDK_ROOT"
echo "  Output:     $OUTPUT_DIR"
echo "  Graph:      $GRAPH_NAME"
if [ "$DO_ONNX_SURGERY" -eq 1 ]; then
    echo "  ONNX:       $INPUT_ONNX  →  (split outputs)"
else
    echo "  ONNX:       ${FINAL_READY_ONNX:-<will be created>}"
fi
if [ "$DO_CALIB_GEN" -eq 1 ]; then
    echo "  Calib:      $CALIB_DIR  →  (generate raw, $CALIB_NUM images)"
else
    echo "  Calib:      ${FINAL_INPUT_LIST:-<will be created>}"
fi
echo "============================================================"

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: ONNX surgery — split [1,84,8400] into boxes + scores
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_ONNX_SURGERY" -eq 1 ]; then
    echo ""
    echo "--- Step 1: ONNX surgery ---"
    FINAL_READY_ONNX="$WORK_DIR/best_ready.onnx"

    NUM_CLASSES_ARG=()
    [ -n "$NUM_CLASSES" ] && NUM_CLASSES_ARG=(--num-classes "$NUM_CLASSES")

    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_onnx.py" \
        --input "$INPUT_ONNX" \
        --output "$FINAL_READY_ONNX" \
        "${NUM_CLASSES_ARG[@]}"

    echo "  Split ONNX: $FINAL_READY_ONNX"
else
    echo ""
    echo "--- Step 1: ONNX surgery (skipped) ---"
    echo "  Using: $FINAL_READY_ONNX"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Calibration data generation
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_CALIB_GEN" -eq 1 ]; then
    echo ""
    echo "--- Step 2: Calibration data ---"
    FINAL_INPUT_LIST="$WORK_DIR/calib_raw/input_list.txt"

    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_calib.py" \
        --images "$CALIB_DIR/*.jpg" \
        --num "$CALIB_NUM" \
        --output "$(dirname "$FINAL_INPUT_LIST")"

    RAW_COUNT=$(ls "$(dirname "$FINAL_INPUT_LIST")"/*.raw 2>/dev/null | wc -l)
    echo "  Input list: $FINAL_INPUT_LIST"
    echo "  Raw files:  $RAW_COUNT"
else
    echo ""
    echo "--- Step 2: Calibration (skipped) ---"
    echo "  Using: $FINAL_INPUT_LIST"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3: ONNX → unquantized DLC
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Step 3: ONNX → unquantized DLC ---"
UNQUANT_DLC="$WORK_DIR/${GRAPH_NAME}.dlc"

# Auto-detect the input tensor name from the ONNX graph
INPUT_NAME=$("$PYTHON_BIN" -c "
import onnx
m = onnx.load('$FINAL_READY_ONNX')
print(m.graph.input[0].name)
" 2>/dev/null || echo "images")

echo "  Input: $INPUT_NAME  shape: $INPUT_SHAPE"

"$QAIRT_CONVERTER" \
    --input_network "$FINAL_READY_ONNX" \
    --output_path "$UNQUANT_DLC" \
    --source_model_input_shape "$INPUT_NAME" "$INPUT_SHAPE" \
    --target_backend HTP

DLC_SIZE=$(stat -c%s "$UNQUANT_DLC" 2>/dev/null || stat -f%z "$UNQUANT_DLC" 2>/dev/null)
echo "  DLC: $UNQUANT_DLC  ($(( DLC_SIZE / 1024 / 1024 )) MB)"

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4: Quantize DLC (int8)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Step 4: Quantize DLC (int8) ---"
QUANT_DLC="$WORK_DIR/${GRAPH_NAME}_quant.dlc"

echo "  Running quantizer with $CALIB_NUM calibration samples..."
"$QAIRT_QUANTIZER" \
    --input_dlc "$UNQUANT_DLC" \
    --output_dlc "$QUANT_DLC" \
    --input_list "$FINAL_INPUT_LIST" \
    --act_quantizer_calibration min-max \
    --param_quantizer_calibration min-max \
    --target_backend HTP

echo "  Quantized DLC: $QUANT_DLC"

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5: DLC → context binary (.bin)
#
#  Reads SoC settings from the TWO-FILE backend config in the project root:
#    config_file.json          ← wrapper passed to --config_file
#      └─ points to → htp_backend_extensions.json
#                       ├── graphs[].vtcm_mb
#                       └── devices[].soc_id, dsp_arch
#
#  QCS6490 defaults are in the repo. Edit for your hardware.
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Step 5: DLC → context binary ---"

BIN_PATH="$OUTPUT_DIR/yolov8_q6a.bin"
ROOT_WRAPPER="$SCRIPT_DIR/config_file.json"
ROOT_BACKEND="$SCRIPT_DIR/htp_backend_extensions.json"

if [ "$SKIP_BINARY_GEN" -eq 1 ]; then
    echo "  SKIPPED: --skip-binary-gen flag set."
    echo ""
    echo "  Generate on-device:"
    echo "    scp $QUANT_DLC radxa@<ip>:/tmp/"
    echo "    scp generate_binary_ondevice.sh radxa@<ip>:/tmp/"
    echo "    ssh radxa@<ip> 'cd /tmp && ./generate_binary_ondevice.sh /tmp/${GRAPH_NAME}_quant.dlc'"
    echo "    scp radxa@<ip>:/tmp/export/yolov8_q6a.bin $OUTPUT_DIR/"
    echo ""
elif [ -f "$ROOT_WRAPPER" ] && [ -f "$ROOT_BACKEND" ]; then
    echo "  Backend config: $ROOT_WRAPPER → $ROOT_BACKEND"
    "$BIN_GEN" \
        --dlc_path "$QUANT_DLC" \
        --output_dir "$OUTPUT_DIR" \
        --binary_file "yolov8_q6a" \
        --backend "$HTP_BACKEND" \
        --config_file "$ROOT_WRAPPER"
else
    echo "  ERROR: Backend configs not found."
    echo "    Expected: $ROOT_WRAPPER"
    echo "    Expected: $ROOT_BACKEND"
    echo "  These files ship with the repo. Restore them and retry."
    exit 1
fi

if [ -f "$BIN_PATH" ]; then
    BIN_SIZE=$(stat -c%s "$BIN_PATH" 2>/dev/null || stat -f%z "$BIN_PATH" 2>/dev/null)
    echo "  Binary: $BIN_PATH  ($(( BIN_SIZE / 1024 )) KB)"
else
    echo "  Note: No binary found (expected with --skip-binary-gen)."
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6: Extract quantization encodings & generate config JSON
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Step 6: Config JSON ---"

ENCODING_JSON="$WORK_DIR/${GRAPH_NAME}_encodings.json"
if [ -x "$DLC_TO_JSON" ]; then
    "$DLC_TO_JSON" --input_dlc "$QUANT_DLC" --output_json "$ENCODING_JSON" 2>/dev/null || true
fi

export ENCODING_JSON DLC_INFO QUANT_DLC
ENCODING_DATA=$("$PYTHON_BIN" -c "
import json, os, sys, subprocess, re

json_path = os.environ.get('ENCODING_JSON', '')
dlc_info_cmd = os.environ.get('DLC_INFO', 'qairt-dlc-info')
quant_dlc = os.environ.get('QUANT_DLC', '')

encodings = {}

if json_path and os.path.exists(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        graph = data.get('graph', data)
        tensors = graph.get('tensors', [])
        for t in tensors:
            ttype = t.get('tensor_type', '')
            name = t.get('name', '')
            enc_list = t.get('encoding', [])
            if isinstance(enc_list, dict): enc_list = [enc_list]
            if not enc_list: enc_list = [{}]
            enc = enc_list[0] if enc_list else {}
            if ttype == 'APP_WRITE':
                encodings['input_name'] = name
                encodings['input_scale'] = enc.get('scale', 0.0)
                encodings['input_offset'] = enc.get('offset', 0)
            elif ttype == 'APP_READ':
                dims = t.get('dimensions', [])
                if len(dims) == 3 and dims[1] == 4:
                    encodings['boxes_name'] = name
                    encodings['boxes_scale'] = enc.get('scale', 0.0)
                    encodings['boxes_offset'] = enc.get('offset', 0)
                elif len(dims) == 3 and dims[1] != 4:
                    encodings['scores_name'] = name
                    encodings['scores_scale'] = enc.get('scale', 0.0)
                    encodings['scores_offset'] = enc.get('offset', 0)
    except Exception:
        pass

if not encodings.get('boxes_scale') or not encodings.get('scores_scale'):
    try:
        result = subprocess.run(
            [dlc_info_cmd, '--input_dlc', quant_dlc, '--display_all_encodings'],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout + result.stderr
        # ── Auto-detect encoding lines for any tensor ─────────────
        encoding_pattern = re.compile(
            r'(\S+)\s+encoding\s*:\s*bitwidth\s+\d+,\s*min\s+([\d.e+-]+),\s*max\s+([\d.e+-]+),\s*scale\s+([\d.e+-]+),\s*offset\s+([\d.e+-]+)'
        )
        for m in encoding_pattern.finditer(text):
            tname = m.group(1)
            scale = float(m.group(4))
            offset = float(m.group(5))
            if tname == 'images':
                encodings['input_name'] = tname
                encodings['input_scale'] = scale
                encodings['input_offset'] = offset
            elif 'Sigmoid' in tname:
                encodings['scores_name'] = tname
                encodings['scores_scale'] = scale
                encodings['scores_offset'] = offset
            elif 'Mul_5' in tname or 'Mul_' in tname:
                # Pick the last Mul_* output (boxes)
                encodings['boxes_name'] = tname
                encodings['boxes_scale'] = scale
                encodings['boxes_offset'] = offset
    except Exception:
        pass

bs = encodings.get('boxes_scale', 0) or 0
ss = encodings.get('scores_scale', 0) or 0
is_ = encodings.get('input_scale', 0) or 0
print(f'INPUT_SCALE={is_:.15g}')
print(f'INPUT_OFFSET={encodings.get(\"input_offset\", 0)}')
print(f'INPUT_NAME={encodings.get(\"input_name\", \"images\")}')
print(f'BOXES_SCALE={bs:.15g}')
print(f'BOXES_OFFSET={encodings.get(\"boxes_offset\", 0)}')
print(f'BOXES_NAME={encodings.get(\"boxes_name\", \"output_0\")}')
print(f'SCORES_SCALE={ss:.15g}')
print(f'SCORES_OFFSET={encodings.get(\"scores_offset\", 0)}')
print(f'SCORES_NAME={encodings.get(\"scores_name\", \"output_1\")}')
")

eval "$ENCODING_DATA" 2>/dev/null || {
    echo "  WARNING: Could not parse encodings, using defaults."
    INPUT_SCALE=0.003921568859; INPUT_OFFSET=0.0; INPUT_NAME="images"
    BOXES_SCALE=2.556329488754; BOXES_OFFSET=0.0; BOXES_NAME="output_0"
    SCORES_SCALE=0.003800418461; SCORES_OFFSET=0.0; SCORES_NAME="output_1"
}

# Determine num_classes for config JSON
if [ -n "$NUM_CLASSES" ]; then
    NUM_CLASSES_FOR_JSON="$NUM_CLASSES"
else
    NUM_CLASSES_FOR_JSON=80  # safe default for standard YOLOv8
fi

echo "  Input:  $INPUT_NAME  scale=$INPUT_SCALE  offset=$INPUT_OFFSET"
echo "  Boxes:  $BOXES_NAME  scale=$BOXES_SCALE  offset=$BOXES_OFFSET"
echo "  Scores: $SCORES_NAME  scale=$SCORES_SCALE  offset=$SCORES_OFFSET"

DETECTED_GRAPH="$GRAPH_NAME"
if DLC_INFO_OUT="$("$DLC_INFO" --input_dlc "$QUANT_DLC" 2>/dev/null)"; then
    AUTO_GRAPH=$(echo "$DLC_INFO_OUT" | grep -oP 'Info of graph:\s*\K\S+' | head -1 || true)
    [ -n "$AUTO_GRAPH" ] && DETECTED_GRAPH="$AUTO_GRAPH"
fi

SDK_VERSION=$(grep "^version:" "$SDK_ROOT/sdk.yaml" 2>/dev/null | awk '{print $2}' || echo "2.45.40")

CONFIG_PATH="$OUTPUT_DIR/yolov8_q6a_config.json"
cat > "$CONFIG_PATH" << JSONEOF
{
  "model_file": "yolov8_q6a.bin",
  "graph_name": "$DETECTED_GRAPH",
  "sdk_version": "$SDK_VERSION",
  "input_spec": "{None: [TensorSpec(name='$INPUT_NAME', dtype='uint8', shape=(1, 3, 640, 640), scale=$INPUT_SCALE, zero_point=${INPUT_OFFSET})]}",
  "output_spec": "{None: [TensorSpec(name='$SCORES_NAME', dtype='uint8', shape=(1, $NUM_CLASSES_FOR_JSON, 8400), scale=$SCORES_SCALE, zero_point=${SCORES_OFFSET}), TensorSpec(name='$BOXES_NAME', dtype='uint8', shape=(1, 4, 8400), scale=$BOXES_SCALE, zero_point=${BOXES_OFFSET})]}",
  "boxes_scale": $BOXES_SCALE,
  "scores_scale": $SCORES_SCALE,
  "input_scale": $INPUT_SCALE,
  "boxes_name": "$BOXES_NAME",
  "scores_name": "$SCORES_NAME",
  "input_name": "$INPUT_NAME"
}
JSONEOF

echo "  Config: $CONFIG_PATH"
echo "  Graph:  $DETECTED_GRAPH"

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
echo ""
echo "  Output files:"
ls -lh "$OUTPUT_DIR"/yolov8_q6a_config.json 2>/dev/null
if [ -f "$BIN_PATH" ]; then
    ls -lh "$BIN_PATH" 2>/dev/null
    echo ""
    echo "  Deploy to device:"
    echo "    scp $BIN_PATH radxa@<ip>:/home/radxa/src/exported_model/quantized_compiled_model/"
    echo "    scp $CONFIG_PATH radxa@<ip>:/home/radxa/src/exported_model/quantized_compiled_model/"
else
    echo "  (no binary — use generate_binary_ondevice.sh on the device)"
fi
echo ""
echo "  On-device inference (after deploying .bin):"
echo "    qnn-net-run --backend lib/libQnnHtp.so \\"
echo "        --retrieve_context yolov8_q6a.bin \\"
echo "        --input_list input_list.txt \\"
echo "        --output_dir ./output"
echo ""
echo "  Intermediate files (safe to delete):"
echo "    rm -rf $WORK_DIR"
echo ""
