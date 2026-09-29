# Data Collection & Annotation Guide

This document centralizes all dataset collection, filtering, and annotation instructions.

## Dataset Strategy

### Source Scope

This project currently uses five sources:

1. **AVA Dataset** for Level 1 to Level 3:
   - `level1_poor/` -> AVA score <= 4.5
   - `level2_acceptable/` -> AVA score 5.0–7.5
   - `level3_good/` -> AVA score 7.5–9.0
2. **FFHQ Dataset (Hugging Face)** for Level 3:
   - downloaded via `huggingface_hub`
   - raw: `data/raw/level3_good/ffhq/`
   - processed: `data/processed/level3_good/`
   - fixed score `8.0` (`annotator: ffhq_fixed`)
3. **Pexels API** for Level 4:
   - fixed score `9.5`
4. **Unsplash API** for Level 4:
   - fixed score `9.5`
   - download prefilter: `width >= 800` and `height >= 800`
5. **SmugMug ImageSearch API** for Level 4:
   - fixed score `9.5`
   - requires `SMUGMUG_API_KEY`

AVA is the primary source because it provides large-scale human aesthetic preference ratings.

### AVA vote-distribution filtering

- Keep as-is when `score_std < 1.0`.
- Keep with smoothing when `1.0 <= score_std < 1.5`:
  - `smoothed_score = raw_score * 0.9 + 5.0 * 0.1`
- Discard when `score_std >= 1.5`.

`score_std` is stored in annotations next to `score`.

## AVA Collection (Two-stage)

Run in project root:

```bash
cd ~/Humanframe-AI
source .venv/bin/activate
```

Stage A (download raw only):

```bash
python3 data_collect/ava_download.py \
  --mode download_raw \
  --clean-levels \
  --clean-annotations \
  --save-every 200
```

Check Stage A:

```bash
ls -lh portraiq/data/annotations/ava_raw_download.json
```

Stage B (raw -> processed with person filtering):

```bash
python3 data_collect/ava_download.py \
  --mode build_processed \
  --person-filter \
  --person-backend torchvision \
  --person-device cuda:0 \
  --person-imgsz 384 \
  --throttle-ms 20 \
  --clean-levels \
  --clean-annotations \
  --save-every 200
```

Resume Stage B (interrupted run):

```bash
python3 data_collect/ava_download.py \
  --mode build_processed \
  --person-filter \
  --person-backend torchvision \
  --person-device cuda:0 \
  --person-imgsz 384 \
  --throttle-ms 20 \
  --resume \
  --save-every 200
```

## Level 3 Collection (FFHQ)

```bash
cd ~/Humanframe-AI
source .venv/bin/activate
python3 data_collect/ffhq_download.py \
  --max_images 70000 \
  --person_conf 0.25 \
  --resume
```

Optional explicit repo:

```bash
python3 data_collect/ffhq_download.py \
  --repo_id marcosv/ffhq-dataset \
  --max_images 70000 \
  --person_conf 0.25 \
  --resume
```

This writes:
- Raw images: `portraiq/data/raw/level3_good/ffhq/`
- Processed images: `portraiq/data/processed/level3_good/`
- Annotations: `portraiq/data/annotations/level3_ffhq.json`

## Level 4 Collection (Pexels + Unsplash)

Set keys:

```bash
export PEXELS_API_KEY="<your_pexels_api_key>"
export UNSPLASH_ACCESS_KEY="<your_unsplash_access_key>"
```

Foreground:

```bash
python3 data_collect/level4_download.py --source all --max_images 5000 --resume
python3 data_collect/level4_download.py --source pexels --max_images 5000 --resume
python3 data_collect/level4_download.py --source unsplash --max_images 1000 --resume
```

Background:

```bash
mkdir -p logs
nohup python3 data_collect/level4_download.py \
  --source all \
  --max_images 5000 \
  --per_keyword 2000 \
  --person_conf 0.25 \
  --resume \
  > logs/level4_download.log 2>&1 &
echo $! > logs/level4_download.pid
```

Monitor/stop:

```bash
tail -f logs/level4_download.log
kill "$(cat logs/level4_download.pid)"
```

This writes:
- Raw images: `portraiq/data/raw/level4_excellent/`
- Processed images: `portraiq/data/processed/level4_excellent/`
- Annotations: `portraiq/data/annotations/level4_excellent.json`

## Level 4 Collection (SmugMug)

Set key:

```bash
export SMUGMUG_API_KEY="your_key_here"
```

```bash
python3 data_collect/smugmug_download.py \
  --max_images 5000 \
  --per_keyword 1500 \
  --scope /api/v2/user/cmac \
  --discover_scopes \
  --max_scopes 50 \
  --person_conf 0.25 \
  --resume
```

This writes:
- Raw images: `portraiq/data/raw/level4_excellent/`
- Processed images: `portraiq/data/processed/level4_excellent/`
- Annotations: `portraiq/data/annotations/level4_smugmug.json`

## Annotation JSON Format

```json
{
  "image_id": "img_0042",
  "filename": "level2_acceptable/portrait_042.jpg",
  "category": "portrait",
  "score": 6.3,
  "score_std": 0.82,
  "annotator": "ava",
  "split": "train"
}
```

## Data Troubleshooting (`total: 0`)

If your training split check prints `total: 0 train: 0 val: 0 test: 0`, verify:

1. JSON files exist under `data/annotations/`.
2. You are in the correct working directory.
3. JSON structure contains `filename` and `score`.
4. If using explicit split labels, some records must have `"split": "train"`.
5. Referenced image files exist under `data/processed/`.
