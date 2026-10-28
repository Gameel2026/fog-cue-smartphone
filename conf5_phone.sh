#!/data/data/com.termux/files/usr/bin/bash
set -e
cd "$(dirname "$0")"
MODE=$1; TAG=${2:-main}

case "$MODE" in
  build)
    pkg list-installed 2>/dev/null | grep -q '^clang/' || pkg install -y clang
    clang -O2 -o fog_edge_c fog_edge_c.c -lm
    python conf5_export.py
    echo "built: $(./fog_edge_c 2>&1 | head -1)"; clang --version | head -1 > c5_${TAG}_compiler.txt ;;
  bench)
    for v in lab_fp32 lab_int8 home_fp32 home_int8; do
      case $v in lab_*) rp=c_replay_lab.bin; npz=edge_model_${v#lab_}.npz ;; home_*) rp=c_replay_home.bin; npz=edge_home_${v#home_}.npz ;; esac
      ./fog_edge_c bench c_model_$v.bin $rp c5_${TAG}_$v 10
      python conf5_summarize.py bench c5_${TAG}_$v $npz
    done ;;
  dataset)
    ./fog_edge_c dataset c_model_lab_fp32.bin  c_streams_defog.bin    c5_dataset_defog_lab_fp32.csv
    ./fog_edge_c dataset c_model_lab_int8.bin  c_streams_defog.bin    c5_dataset_defog_lab_int8.csv
    ./fog_edge_c dataset c_model_home_fp32.bin c_streams_defog.bin    c5_dataset_defog_home_fp32.csv
    ./fog_edge_c dataset c_model_home_int8.bin c_streams_defog.bin    c5_dataset_defog_home_int8.csv
    ./fog_edge_c dataset c_model_lab_fp32.bin  c_streams_FoG-STAR.bin c5_dataset_fogstar_lab_fp32.csv
    ./fog_edge_c dataset c_model_lab_int8.bin  c_streams_FoG-STAR.bin c5_dataset_fogstar_lab_int8.csv
    for v in lab_fp32 lab_int8 home_fp32 home_int8; do python conf5_summarize.py dataset c_refs_defog.npz c5_dataset_defog_$v.csv $v; done
    for v in lab_fp32 lab_int8; do python conf5_summarize.py dataset c_refs_FoG-STAR.npz c5_dataset_fogstar_$v.csv $v; done
    python conf5_summarize.py dataset c_refs_defog.npz c5_dataset_defog_lab_fp32.csv torch
    python conf5_summarize.py dataset c_refs_FoG-STAR.npz c5_dataset_fogstar_lab_fp32.csv torch ;;
  realtime)
    M=${3:-lab}; termux-wake-lock || true
    ./fog_edge_c realtime c_model_${M}_int8.bin c_replay_${M}.bin 60 c5_${TAG}_${M}_screen
    python conf5_summarize.py realtime c5_${TAG}_${M}_screen; termux-wake-unlock || true ;;
  screenoff)
    termux-wake-lock || true
    echo "Turn the screen OFF now. The run starts in 30 s and lasts 60 min."; sleep 30
    ./fog_edge_c realtime c_model_lab_int8.bin c_replay_lab.bin 60 c5_${TAG}_lab_screenoff
    python conf5_summarize.py realtime c5_${TAG}_lab_screenoff; termux-wake-unlock || true ;;
  stream)
    termux-wake-lock || true
    ./fog_edge_c stream c_model_lab_int8.bin ${3:-listen} c5_${TAG}_lab
    python conf5_summarize.py stream c5_${TAG}_lab edge_model_int8.npz c_model_lab_int8.bin replay_defog.npz
    termux-wake-unlock || true ;;
  *) echo "usage: bash conf5_phone.sh build|bench TAG|dataset|realtime TAG MODEL|screenoff TAG|stream TAG [LAPTOP_IP]"; exit 2 ;;
esac
