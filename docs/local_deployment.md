# Local Deployment On 3080Ti

This phase is a dry-run engineering MVP. It does not train Qwen2.5-VL and does not download model weights.

```bash
bash scripts/00_check_env.sh
bash scripts/01_install_env.sh
bash scripts/02_make_seed_data.sh
python src/data/validate_dataset.py --input data/processed/drivemind_seed.jsonl
bash scripts/03_convert_data.sh
bash scripts/04_run_dry_infer.sh
bash scripts/05_run_eval.sh
bash scripts/06_run_safety_guard.sh
python src/rewards/total_reward.py --demo
bash scripts/07_run_demo.sh
```

Use `--smoke_test` for a non-blocking demo check:

```bash
bash scripts/07_run_demo.sh --smoke_test
```

On Windows, if `bash` is not on PATH, use Git Bash directly:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/00_check_env.sh
```

For the local MVP and demo only, install the lighter dependency set:

```bash
bash scripts/01_install_env.sh --mvp-only
```

If the active Anaconda environment is not writable, use user-site install:

```bash
bash scripts/01_install_env.sh --mvp-only --user
```

For full model inference preparation, use:

```bash
bash scripts/01_install_env.sh
```

If PyTorch fails with a `typing_extensions` import error, upgrade `typing_extensions` in the active environment:

```bash
pip install -U typing_extensions
```

Optional Qwen2.5-VL inference entrypoint defaults to dry-run:

```bash
bash scripts/04_run_qwen25vl_infer.sh --dry_run --max_samples 5
```

Real inference requires a local model directory:

```bash
bash scripts/04_run_qwen25vl_infer.sh \
  --model_name_or_path /path/to/Qwen2.5-VL-3B-Instruct \
  --max_samples 5 \
  --load_in_4bit
```
