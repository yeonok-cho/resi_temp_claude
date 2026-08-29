# side_alarm — RESI / TEMP 사이드 비대칭 경보

칩 단위로 이미 계산되어 DB에 쌓인 두 값을 소비한다. 어느 쪽도 재계산하지 않는다.

| 값 | 컬럼 | 기존 정의 |
|---|---|---|
| RESI anomaly score | `mp_score` | `rmt_profile_detector.BaseProfile.score()` |
| TEMP roughness | `temp_roughness` | `feature_extractor.compute_roughness` = `std(diff(residual))` |

`module_id`가 `left` / `right` 값을 담고 있어 SIDE로 정규화된다.

```
chip rows (mp_score, temp_roughness)
        │
        ├── RESI detector ──► RESI_LEVEL / RESI_EVENT / RESI_REASON
        │
        └── TEMP detector ──► TEMP_LEVEL / TEMP_EVENT / TEMP_REASON
                                    │
                              equipment trend
                                    │
                              severity fusion ──► FINAL_SEVERITY
```

RESI 경보에 TEMP는 필요하지 않고, TEMP 경보에 RESI도 필요하지 않다. fusion은
둘이 함께 이상일 때 **올리기만** 하며, 두 판정 모두 최종 행에 그대로 남는다.

## 파일

| 파일 | 스펙 조항 |
|---|---|
| `config.py` | §22 — 모든 임계값 중앙화 |
| `loader.py` | §3 입력 키, SIDE 정규화, §25 최신 측정, §32 groupby 집계 |
| `baseline.py` | §5 RESI · §9 TEMP 장비-사이드 baseline, §14 양자화 진단, §24 누수 방지 |
| `detect_resi.py` | §4 통계·비대칭, §6 등급, §7 PROCESS_SHIFT |
| `detect_temp.py` | §10 high share, §11 suspect, §13 legacy, §8.1 RESI 로직 미사용 |
| `trend.py` | §12 WAFER_EVENT / TRANSIENT / EQUIPMENT_TREND |
| `fuse_alarm.py` | §16–§19 fusion matrix, §20 출력 스키마 |
| `scripts/run_side_alarm.py` | §29 로깅, §30 출력 파일 |
| `tests/test_side_alarm.py` | §28 Case A–G |

## 실행

```bash
# ClickHouse (CLICKHOUSE_HOST / _USER / _PASSWORD 환경변수)
python side_alarm/scripts/run_side_alarm.py \
    --start '2026-08-01 00:00:00' --end '2026-08-29 00:00:00' \
    --baseline-end '2026-08-22 00:00:00' \
    --output-dir /workspace/act_resi/sw_act_resi_anomaly/output

# 칩 단위 CSV (DB 없이)
python side_alarm/scripts/run_side_alarm.py --input-csv chips.csv \
    --baseline-end '2026-08-22 00:00:00' --output-dir ./out
```

출력: `resi_anomaly_result.csv`, `temp_anomaly_result.csv`,
`final_alarm_result.csv`, `equipment_resi_baseline.csv`,
`equipment_temp_baseline.csv`.

## 누수 방지

baseline은 `--baseline-end` 직전 `--baseline-lookback-days` 구간에서만 만들고,
그 시각 이후 wafer만 채점한다. 따라서 시각 t의 wafer는 항상 t보다 엄격히 오래된
데이터하고만 비교된다. 백테스트는 하나의 baseline을 재사용하지 말고
`--baseline-end`를 앞으로 밀면서 반복한다. trend 판정도 과거만 보므로
`ONLINE_RESULT = True`로 표시된다.

## 알아둘 점

- **SD 카운트 기준**: 3/5/7/9 시그마 카운트는 기본적으로 장비-사이드 baseline의
  칩 분포(`RESI_CHIP_MEAN/STD`) 대비로 센다. 웨이퍼 자체 std를 쓰면 고장난 히터가
  자기 std를 부풀려 자기 카운트를 눌러버리기 때문이다. `ResiConfig.sd_reference`로
  바꿀 수 있다.
- **양자화**: TEMP는 std ratio를 1차 판정에 쓰지 않는다. 양자화된 장비에서
  사이드 std가 0이 되어 비율이 무한대로 튀기 때문이다. 대신 사이드별 P99 초과
  칩 비율을 쓰고, legacy std-ratio 경로는 양자화가 아닌 장비에서만 켜진다.
- **최소 샘플**: 칩 수가 부족하면 조용히 NORMAL로 두지 않고
  `INSUFFICIENT_DATA`로 표시한다.
- **편측 결측**: 한쪽 사이드만 있으면 `SIDE_COMPARISON_AVAILABLE=False`가 되고
  비대칭 판정은 하지 않는다. 절대값 baseline 검사는 계속 돈다.
