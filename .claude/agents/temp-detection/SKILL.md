---
name: temp-detection
description: >
  온도 진동 이상감지 파이프라인의 모델(inference) 단계 가이드.
  전처리에서 나온 roughness를 입력받아 두 개의 탐지기를 실행한다:
  한 웨이퍼 안에서의 drift를 잡는 CUSUM(intra-wafer)과
  여러 웨이퍼에 걸친 느린 drift를 잡는 EWMA 제어도(inter-wafer).
  "CUSUM", "EWMA", "intra", "inter", "탐지기", "drift 감지",
  "이상감지 로직", "control chart", "inference" 등의 맥락에서 참조하라.
---

# 탐지 모델 (Detection / Inference)

전처리(temp-preprocessing)에서 만든 roughness를 받아
**두 개의 독립 탐지기**를 실행한다. 학습(fitting) 없이 통계적 파라미터만으로
동작하므로 "inference only" 구조다.

---

## 탐지기 A — Intra-wafer CUSUM

**목적**: 한 웨이퍼 안에서 bond 순서가 진행될수록 roughness가 누적 증가하는지 탐지.

### 동작 원리

```
1. warmup: 첫 warmup_chips(10)개 칩으로 이 웨이퍼의 정상 수준 추정
   warmup_mean = mean(roughness[:10])
   effective_std = max(std(roughness[:10]), warmup_mean × 0.15, 1e-9)
   ↑ std가 너무 작을 때 과민해지지 않도록 floor 적용

2. z-score: 모든 칩을 warmup 기준으로 표준화
   z[i] = (roughness[i] - warmup_mean) / effective_std

3. CUSUM (one-sided, upward):
   S[i] = max(0, S[i-1] + z[i] - k)      k = cusum_k = 0.5

4. alarm: S[i] > h (cusum_h = 4.0)이면 경보
```

- `k = 0.5`: 이 정도 차이(0.5σ)까지는 봐주는 허용치. 1σ shift를 최적으로 탐지하도록 설계된 값.
- `h = 4.0`: 누적합이 이 값을 넘으면 경보. 낮을수록 민감, 높을수록 false alarm 감소.
- CUSUM은 한 번 넘으면 계속 누적되므로, 조기 발견 시점은 처음 h를 넘는 타이밍.
- 코드: `temp_anomaly/temp_vibration_detection/detector/intra_wafer.py`

### 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `cusum_k` | 0.5 | 허용치 (allowance). 낮을수록 작은 drift에 민감 |
| `cusum_h` | 4.0 | 경보 임계값 (decision interval). 낮을수록 빠른 경보 |
| `warmup_chips` | 10 | 웨이퍼 내부 baseline 구간 칩 수 |

### 특성

- **self-calibrating**: 매 웨이퍼마다 독립적으로 warmup baseline을 재계산하므로
  전역 baseline 없이 첫 번째 이상 웨이퍼부터 작동한다.
- PM 이벤트에 영향받지 않음 (baseline이 웨이퍼마다 리셋되니까).
- 웨이퍼가 최소 `warmup_chips × 2 = 20`개 칩 미만이면 탐지 건너뜀.

---

## 탐지기 B — Inter-wafer EWMA 제어도

**목적**: 여러 웨이퍼에 걸쳐 평균 roughness가 서서히 올라가는 drift 탐지.

### 동작 원리

```
1. baseline 구축 (첫 baseline_wafers = 10장):
   Welford 온라인 알고리즘으로 baseline_mean, baseline_std 누적
   → 10장 후 동결(freeze) — 이후 drift가 와도 baseline이 따라 올라가지 않음

2. EWMA 추적 (매 웨이퍼):
   ewma = (1 - λ) × ewma + λ × wafer_mean_roughness     λ = ewma_lambda = 0.1

3. alarm (baseline 동결 후에만):
   control_limit = baseline_mean + alert_sigma × baseline_std
   ewma > control_limit 이면 경보
```

- `λ = 0.1`: 새 웨이퍼 10% 반영 → 반감기 약 6.6웨이퍼, 95% 수렴 약 28웨이퍼
- `alert_sigma = 3.0`: 3σ 초과 시 경보 (정상 분포 가정 시 false alarm률 약 0.13%)
- baseline이 동결되므로 실제 drift 발생 시에도 기준이 같이 올라가지 않아 탐지 가능
- 코드: `temp_anomaly/temp_vibration_detection/detector/inter_wafer.py`

### 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `ewma_lambda` | 0.1 | EWMA 반영률. 낮을수록 느린 drift에 민감 |
| `alert_sigma` | 3.0 | 경보 기준 (시그마 단위). 낮을수록 민감 |
| `baseline_wafers` | 10 | baseline 구축에 사용하는 웨이퍼 수 |

### 특성

- baseline 구축 기간(10장)에는 경보 없음.
- baseline 동결 이후에도 EWMA만 계속 업데이트됨.
- PM 이벤트 후 `pipeline.reset_baseline(group_key)` 호출 필요 — 리셋 시 baseline 재구축 시작.

---

## 두 탐지기의 관계

```
roughness (전처리 출력)
       │
       ├─► Intra-wafer CUSUM ─► intra_drift 이벤트 (없으면 빈 리스트)
       │
       └─► Inter-wafer EWMA ─► inter_drift 이벤트 (없으면 빈 리스트)
```

- 완전히 독립적으로 실행되며 서로 영향을 주지 않는다.
- 같은 웨이퍼에서 둘 다 경보가 날 수 있다.
- severity 단위가 달라서 직접 비교/합산하지 않는다.

---

## 재현 방법

```python
from temp_anomaly.temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_anomaly.temp_vibration_detection.config import DEFAULT_CONFIG

pipeline = TempVibrationPipeline(config=DEFAULT_CONFIG)

# baseline 사전 구축 (known-good 웨이퍼로)
pipeline.initialize_baseline(group_key, good_wafers)

# 이후 매 웨이퍼 처리
for wafer in production_wafers:
    events = pipeline.process_wafer(wafer)
    for e in events:
        print(e)   # [INTRA_DRIFT] or [INTER_DRIFT] + severity
```

baseline 저장/복원:
```python
pipeline.save_baseline("state/baseline.json")
pipeline.load_baseline("state/baseline.json")
```

시각화 예시: `temp_anomaly/examples/drift_demo.py`

판정 해석: **temp-verdict** skill 참조.
