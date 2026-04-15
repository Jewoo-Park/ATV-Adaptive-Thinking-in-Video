## GRPO_Video 벤치 테스트 실행 가이드 (재현 가능한 세팅)

이 문서는 클러스터(PBS `qsub -I`) 환경에서 **3개 모델 × 3개 벤치(UVB/VideoMMMU/MMVU)** 를 안정적으로 실행하기 위한 “재현 가능한 절차”입니다.  
현재 기준은 **HF(Transformers) backend 강제**로 동작하게 구성되어 있습니다.

---

### 0) 모델 경로(수정)

- **Base**: `/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct`
- **SFT merged**: `/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length`
- **SFT + GRPO merged**: `/scratch/users/ntu/n2500182/models/video_r1_uvb_grpo_answer_only_lora_merged_ckpt1500`

---

### 1) GPU 노드 할당 (PBS interactive)

로그인 노드에서:

```bash
qsub -I -l select=1:ngpus=1:mem=110g -l walltime=20:00:00 -P personal-n2500182 -q normal
```

프롬프트가 `x1000...` 로 바뀌면 GPU 노드입니다.

---

### 2) (중요) 노드마다 Python 런타임 모듈 로드

GPU 노드에서:

```bash
module load python/3.11.7-gcc11
```

이 단계는 노드에 따라 `libpython3.11.so.1.0` 로딩 문제가 생기는 것을 막기 위해 필요합니다.

---

### 3) 벤치용 가상환경(venv) “정확히 재현”하기

GPU 노드에서 레포로 이동 후 아래 스크립트를 실행합니다.

```bash
cd /home/users/ntu/n2500182/workspace_JWP/repos/GRPO_Video
bash src/scripts/setup_bench_venv.sh
```

성공하면 `/home/users/ntu/n2500182/scratch/.venv_bench` 가 준비되고, 설치된 버전이 출력됩니다.

---

### 4) venv 활성화

```bash
source /home/users/ntu/n2500182/scratch/.venv_bench/bin/activate
which python
python -V
```

---

### 5) 벤치 실행 (HF backend 강제)

```bash
cd /home/users/ntu/n2500182/workspace_JWP/repos/GRPO_Video

export MODEL_BASE="/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct"
export MODEL_SFT_MERGED="/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length"
export MODEL_SFT_GRPO_MERGED="/scratch/users/ntu/n2500182/models/video_r1_uvb_grpo_answer_only_lora_merged_ckpt1500"
export PROCESSOR_PATH="$MODEL_BASE"

# 안정적으로 돌리기 위해 vLLM 대신 HF backend 사용
export BENCH_EVAL_EXTRA="--backend hf"

bash src/scripts/run_video_benchmark_matrix.sh
```

---

### 6) 로그 위치/확인 방법

실행마다 아래에 런 폴더가 하나 생깁니다.

- `outputs/video_benchmark_runs/YYYYMMDD_HHMMSS/`

그 안에:

- `base__uvb.log`, `base__videommmu.log`, `base__mmvu.log`, …
- `sft_merged__*.log`, `sft_grpo_merged__*.log`
- `exit_codes.txt` (각 단계 exit code 요약)
- `fix_rope_scaling.log` (모델 config의 rope 관련 자동 정규화 로그)

실시간 확인:

```bash
tail -f outputs/video_benchmark_runs/<타임스탬프>/base__uvb.log
```

---

### 7) 결과(JSON/JSONL) 저장 위치

각 eval 스크립트는 기본으로 **`--model` 디렉터리 아래**에 결과를 저장합니다.

- 예측(샘플별): `*predictions*.jsonl`
- 요약 지표/리포트: `*metrics*.json` 또는 `eval_metrics_*.json`

즉 결과는 다음 3 폴더 하위에 쌓입니다.

- `/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct/`
- `/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length/`
- `/scratch/users/ntu/n2500182/models/video_r1_uvb_grpo_answer_only_lora_merged_ckpt1500/`

---

### 8) 자주 겪는 문제

- **exit 127 / `libpython3.11.so.1.0`**: `module load python/3.11.7-gcc11`을 먼저 하고 venv 활성화하세요.
- **`rope_type mrope`/`type` 충돌**: `run_video_benchmark_matrix.sh`가 시작 시 자동으로 정규화합니다. 런 폴더의 `fix_rope_scaling.log`를 확인하세요.
- **처음에 0%에서 오래 멈춘 듯 보임**: HF backend는 첫 샘플(프레임 로드/첫 generate)이 느릴 수 있습니다. `tail -f`로 로그가 늘어나는지 확인하세요.

