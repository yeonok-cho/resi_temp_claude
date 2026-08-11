---
name: temp-preprocessing
description: >
  온도 진동 이상감지 파이프라인의 전처리 단계 가이드.
  칩별 온도 시계열에서 roughness feature를 추출하는 방법,
  그룹별 reference profile을 EWMA로 유지하는 방법, 그리고
  ClickHouse에서 원본 데이터를 읽어 ChipData / WaferData 구조로
  변환하는 방법을 다룬다.
  "전처리", "roughness", "기준 프로파일", "reference EWMA",
  "temp_series", "잔차", "feature 추출" 등의 맥락에서 참조하라.
---

# 전처리 (Preprocessing)

이 단계는 raw 온도 시계열 → **roughness 스칼라** 변환을 담당한다.
이후 탐지 단계(temp-detection skill)에서 입력으로 쓰인다.

---

## 핵심 개념

### Roughness — 최종 feature 스칼라

```
residual  = chip.temp_series - temp_reference     # 잔차
diff      = residual[i+1] - residual[i]           # 1차 차분
roughness = std(diff)                             # 진동 크기 지표
```

- **온도 레벨(절댓값)이나 느린 추세에 무관**하게, 고주파 떨림(진동)만 잡는다.
- 각 칩마다 roughness 스칼라 1개가 산출된다.
- 코드: `temp_anomaly/temp_vibration_detection/feature_extractor.py:compute_roughness`

### Reference Profile — 그룹별 정상 온도 프로파일

그룹 `(equipment_id, recipe, head)` 별로 "정상 웨이퍼의 온도 모양"을
EWMA로 천천히 추적한다.

```
# 첫 웨이퍼
temp_ref = pointwise_median(이번 웨이퍼 모든 칩의 temp_series)

# 이후 매 웨이퍼
temp_ref = (1 - α) × temp_ref + α × pointwise_median(이번 웨이퍼)
```

- `α = ewma_alpha = 0.1` → 새 웨이퍼 10% 반영, 기존 프로파일 90% 유지
- α=0.1이면 반감기 약 6.6 웨이퍼; 실제 레벨 변화에 약 28장 뒤 수렴
- **detection 전에 업데이트하지 않는다** — pipeline.py가 탐지 후 업데이트하여
  이상 웨이퍼가 기준선을 오염시키는 것을 방지한다
- 코드: `temp_anomaly/temp_vibration_detection/reference_manager.py`

---

## 파라미터 (config.py 기본값)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `ewma_alpha` | 0.1 | reference profile EWMA 반영률 |
| `min_points` | 10 | roughness 계산을 위한 최소 시계열 길이 |

---

## 입출력 스펙

### 입력 (ClickHouse → Python)

테이블: `ds_catalog.aifpa_cow_rule_hist`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `eqp_id` | str | 장비 ID |
| `product` | str | 레시피 |
| `module_id` / `head` | str | 헤드 번호 |
| `wafer_id` | str | 웨이퍼 ID |
| `x`, `y` | int | 칩 좌표 |
| `bond_order` | int | 웨이퍼 내 본딩 순서 (0-based) |
| `temp_raw` | str | hex 인코딩된 float32 온도 시계열 |

hex 디코딩: `_common.py:decode_hex_to_float32`

### 출력

- `ChipFeatures.temp_roughness`: 칩당 roughness 스칼라
- `WaferFeatures`: 한 웨이퍼의 모든 칩 feature 집합
  - `.roughness_sequence()`: bond_order 순 roughness 배열 (intra-wafer CUSUM 입력)
  - `.summary_stats()["mean"]`: 웨이퍼 평균 roughness (inter-wafer EWMA 입력)

---

## 재현 방법

```python
from temp_anomaly.temp_vibration_detection.config import DEFAULT_CONFIG
from temp_anomaly.temp_vibration_detection.reference_manager import ReferenceManager
from temp_anomaly.temp_vibration_detection.feature_extractor import extract_chip_features

ref_mgr = ReferenceManager(DEFAULT_CONFIG.reference)

# 첫 웨이퍼: reference 초기화만 (탐지 없음)
ref_mgr.update(wafer)

# 이후 웨이퍼: feature 추출
ref = ref_mgr.get_or_create(wafer.group_key)
for chip in wafer.chips:
    feat = extract_chip_features(chip, ref.temp_ref, DEFAULT_CONFIG.feature)
    print(chip.wafer_id, chip.bond_order, feat.temp_roughness)
```

시각화 예시: `temp_anomaly/examples/roughness_demo.py`

---

## PM(정비) 이벤트 처리

PM 이후에는 설비 특성이 바뀔 수 있으므로 reference를 리셋해야 한다.

```python
pipeline.reset_baseline(group_key)
# → reference EWMA + inter-wafer EWMA 모두 초기화
# → 다음 웨이퍼부터 reference 재구축 시작
```

다음 단계: **temp-detection** skill 참조.
