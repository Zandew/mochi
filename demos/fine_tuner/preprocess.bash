#! /bin/bash

# Enable job control and set process group
set -eo pipefail
set -x

# Function to display help
usage() {
  echo "Usage: $0 -v|--videos_dir videos_dir -o|--output_dir output_dir -w|--weights_dir weights_dir -n|--num_frames num_frames"
  echo "  -v, --videos_dir            Path to the videos directory"
  echo "  -o, --output_dir            Path to the output directory"
  echo "  -w, --weights_dir           Path to the weights directory"
  echo "  -n, --num_frames            Number of frames"
  echo "  -c, --cpu_only              Run CPU-only preprocessing"
  echo "  -g, --gpu_only              Run GPU-only encoding and embedding"
  exit 1
}

# Function to check if the next argument is missing
check_argument() {
  if [[ -z "$2" || "$2" == -* ]]; then
    echo "Error: Argument for $1 is missing"
    usage
  fi
}

CPU_ONLY=0
GPU_ONLY=0

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -v|--videos_dir) check_argument "$1" "$2"; VIDEOS_DIR="$2"; shift ;;
    -o|--output_dir) check_argument "$1" "$2"; OUTPUT_DIR="$2"; shift ;;
    -w|--weights_dir) check_argument "$1" "$2"; WEIGHTS_DIR="$2"; shift ;;
    -n|--num_frames) check_argument "$1" "$2"; NUM_FRAMES="$2"; shift ;;
    -c|-cpu_only) CPU_ONLY=1;;
    -g|-gpu_only) GPU_ONLY=1;;
    -h|--help) usage ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Check if all required arguments are provided
if [[ -z "$VIDEOS_DIR" || -z "$OUTPUT_DIR" || -z "$WEIGHTS_DIR" || -z "$NUM_FRAMES" ]]; then
  echo "Error: All arguments are required."
  usage
fi

if [[ $CPU_ONLY -eq 1 && $GPU_ONLY -eq 1 ]]; then
  echo "Error: At most one of --cpu_only and --gpu_only can be specified."
  usage
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Using script directory: ${SCRIPT_DIR}"


if [ $GPU_ONLY -eq 0 ]; then
  ##### Step 1: Trim and resize videos
  echo -e "\n\e[1;35m🎬 **Step 1: Trim and resize videos** \e[0m"
  # Calculate duration to trim videos
  DURATION=$(printf "%.1f" "$(echo "($NUM_FRAMES / 30) + 0.09" | bc -l)")
  echo "Trimming videos to duration: ${DURATION} seconds"
  python3 ${SCRIPT_DIR}/trim_and_crop_videos.py ${VIDEOS_DIR} ${OUTPUT_DIR} -d ${DURATION}
fi

if [ $CPU_ONLY -eq 0 ]; then
  ##### Step 2: Run the VAE encoder on each video.
  echo -e "\n\e[1;35m🎥 **Step 2: Run the VAE encoder on each video** \e[0m"
  python3 ${SCRIPT_DIR}/encode_videos.py ${OUTPUT_DIR} \
    --model_dir ${WEIGHTS_DIR} --num_gpus 1 --shape "${NUM_FRAMES}x480x848" --overwrite

  ##### Step 3: Compute T5 embeddings
  echo -e "\n\e[1;35m🧠 **Step 3: Compute T5 embeddings** \e[0m"
  python3 ${SCRIPT_DIR}/embed_captions.py --overwrite ${OUTPUT_DIR}
fi

echo -e "\n\e[1;32m✓ Done!\e[0m"
