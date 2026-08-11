---
name: temp-verdict
description: >
  온도 진동 이상감지 파이프라인의 판정(verdict) 단계 가이드.
  탐지기가 출력한 AnomalyEvent의 severity를 해석하고,
  intra/inter drift 경보를 분류하며, 후속 조치(PM 리셋, 알람 에스컬레이션)
  기준을 정의한다.
  "severity 해석", "판정 기준", "경보 등급", "false alarm",
  "PM 리셋", "이상 분류", "alarm 대응" 등의 맥락에서 참조하라.
---

# 판정 (Verdict)

탐지 단계(temp-detection)에서 나온 `AnomalyEvent` 리스트를 받아
**어느 정도 심각한가, 어떤 종류의 이상인가, 어떤 후속 조치가 필요한가**를 결정한다.

---

## AnomalyEvent 구조

```python
@dataclass
class AnomalyEvent:
    group_key: GroupKey     # (equipment_id, recipe, head)
    wafer_id:  str
    anomaly_type: str       # "intra_drift" | "inter_drift"
    severity:  float        # 탐지 통계량 (단위가 서로 다름)
    details:   dict         # 상세 진단 정보
```

코드: `temp_anomaly/temp_vibration_detection/data_models.py`

---

## Severity 해석 — 탐지기별로 단위가 다르다

### intra_drift severity = CUSUM peak 값

```
severity = max(S)      # CUSUM 누적합의 최댓값 (단위: 무차원)
```

| severity 범위 | 의미 |
|---|---|
| 4.0 ~ 10 | 경보 직후 — 비교적 이른 감지, 웨이퍼 내 drift 초기 단계 |
| 10 ~ 30 | 여러 칩에 걸쳐 지속된 drift — 상당한 누적 |
| 30 이상 | 긴 구간 동안 강한 drift가 지속됨 |

- CUSUM은 경보 후에도 drift가 이어지면 계속 쌓이므로 상한이 없다.
- `details["alarm_chip_index"]`를 보면 **몇 번째 칩부터** 이상이 시작됐는지 알 수 있다.
- `details["n_chips"]`와 비교해 얼마나 빨리 잡았는지 가늠할 수 있다.

### inter_drift severity = 시그마(σ) 거리

```
severity = (ewma_value - baseline_mean) / baseline_std
```

| severity 범위 | 의미 |
|---|---|
| 3.0 ~ 3.5σ | 경보 임계 직후 — 느린 drift의 초기 단계 |
| 3.5 ~ 5σ | 뚜렷한 추세 상승 |
| 5σ 이상 | 기준선 대비 크게 벗어난 상태 |

- 직관적으로 "정상 평균에서 몇 시그마 떨어졌는가"를 나타낸다.
- `details["ewma"]`, `details["baseline_mean"]`, `details["control_limit"]`으로 실제 수치 확인 가능.
- `details["n_baseline_wafers"]`가 `baseline_wafers(10)`보다 작으면 baseline이 아직 불안정함을 의미한다.

---

## 두 경보 유형의 의미 차이

| 구분 | intra_drift | inter_drift |
|---|---|---|
| 탐지 스케일 | 웨이퍼 1장 내부 | 여러 웨이퍼 걸쳐서 |
| 잡는 패턴 | 본딩 진행 중 점진적 악화 | 설비 노화 / 장기 추세 |
| 반응 속도 | 빠름 (당장 이 웨이퍼에서) | 느림 (6~30장 후) |
| PM 필요성 판단 | 단발성 가능 — 재발 여부 확인 필요 | 지속성 높음 — PM 적극 검토 |
| severity 단위 | CUSUM 값 (무차원) | 시그마 (σ) |

**두 severity를 더하거나 평균 내지 않는다** — 단위가 달라 의미없다.

---

## 판정 흐름

```
AnomalyEvent 수신
       │
       ├─ anomaly_type == "intra_drift"
       │       └─► 해당 웨이퍼 재검토 + 동일 그룹 다음 웨이퍼 모니터링
       │           연속 N장 intra_drift → PM 요청 검토
       │
       └─ anomaly_type == "inter_drift"
               └─► 추세 지속 확인 (ewma가 계속 control_limit 초과?)
                   severity > 5σ 또는 여러 장 연속 → PM 즉시 검토
```

---

## PM 이후 처리

PM(예방정비) 완료 후 첫 웨이퍼를 처리할 때 반드시 리셋해야 한다.

```python
pipeline.process_wafer(wafer, pm_event=True)
# 또는
pipeline.reset_baseline(group_key)
```

리셋 효과:
- reference profile EWMA → 초기화 (다음 웨이퍼로 재구축)
- inter-wafer baseline (frozen mean/std, EWMA) → 초기화 및 재구축 시작
- intra-wafer → 영향 없음 (self-calibrating, 웨이퍼마다 자체 baseline)

리셋 없이 PM 후 계속 실행하면, 변경된 설비 상태가 PM 이전 baseline 기준으로
비교되어 false alarm이 과다 발생할 수 있다.

---

## 상태 저장 / 재현

운영 환경에서 파이프라인 상태를 재시작 없이 유지하려면 baseline을 직렬화한다.

```python
# 실행 종료 시
pipeline.save_baseline("state/baseline.json")

# 다음 실행 시 복원
pipeline.load_baseline("state/baseline.json")
```

`baseline.json`에 포함되는 내용:
- 그룹별 `temp_ref` (reference profile EWMA 현재 값)
- 그룹별 inter-wafer `ewma_value`, `baseline_mean`, `baseline_std`, freeze 여부

이 단계까지 완료하면 파이프라인 전 주기가 재현 가능하다.

---

## 관련 파일 빠른 참조

| 단계 | 파일 |
|---|---|
| 전처리 | `temp_anomaly/temp_vibration_detection/feature_extractor.py` |
| reference 관리 | `temp_anomaly/temp_vibration_detection/reference_manager.py` |
| intra-wafer 탐지 | `temp_anomaly/temp_vibration_detection/detector/intra_wafer.py` |
| inter-wafer 탐지 | `temp_anomaly/temp_vibration_detection/detector/inter_wafer.py` |
| 파이프라인 진입점 | `temp_anomaly/temp_vibration_detection/pipeline.py` |
| 파라미터 | `temp_anomaly/temp_vibration_detection/config.py` |
| 데모 (roughness) | `temp_anomaly/examples/roughness_demo.py` |
| 데모 (drift) | `temp_anomaly/examples/drift_demo.py` |
