# Qwen2.5-VL Model Download Notes

Do not store model files inside this repository. Use a server-side model cache such as:

```text
/mnt/models/
  Qwen2.5-VL-3B-Instruct/
  Qwen2.5-VL-7B-Instruct/
```

## Recommended: Hugging Face CLI

```bash
pip install -U "huggingface_hub<1.0"
mkdir -p /mnt/models

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /mnt/models/Qwen2.5-VL-3B-Instruct

huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /mnt/models/Qwen2.5-VL-7B-Instruct
```

If the server has the newer `hf` CLI:

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /mnt/models/Qwen2.5-VL-3B-Instruct

hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /mnt/models/Qwen2.5-VL-7B-Instruct
```

## Recommended In Mainland China: ModelScope

```bash
pip install -U modelscope
mkdir -p /mnt/models

modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /mnt/models/Qwen2.5-VL-3B-Instruct

modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct \
  --local_dir /mnt/models/Qwen2.5-VL-7B-Instruct
```

## Git LFS Alternative

```bash
cd /mnt/models
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

This is simple but less resumable than `huggingface-cli download` or `modelscope download`.

## Sanity Check

```bash
python - <<'PY'
from transformers import AutoProcessor

for path in [
    "/mnt/models/Qwen2.5-VL-3B-Instruct",
    "/mnt/models/Qwen2.5-VL-7B-Instruct",
]:
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    print(path, type(processor).__name__)
PY
```

## DriveMind-VL Smoke Test

After the server environment is ready:

```bash
bash scripts/04_run_qwen25vl_infer.sh \
  --model_name_or_path /mnt/models/Qwen2.5-VL-3B-Instruct \
  --max_samples 5 \
  --load_in_4bit

bash scripts/05_run_eval.sh
```

Run 7B only after 3B smoke test passes.

