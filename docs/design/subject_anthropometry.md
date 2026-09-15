# Subject anthropometry (SMPL fit input)

## Why this exists

[docs/design/dataset_schema.md](dataset_schema.md)와 [docs/status.md](../status.md)에
명시된 대로, 이 저장소는 evidence-backed sequence-to-subject 매핑을 기본적으로
보존하지 않습니다 (`subject_id=null`, `subject_mapping_status=SUBJECT_MAPPING_UNAVAILABLE`).
`dataset_schema.md`는 이 원칙에 명시적 예외를 하나 둡니다 — *"별도 provenance와 함께
제공될 때만 pseudonymous subject ID와 subject-level consensus를 새 version에 추가한다"*.

SMPL fit(`tools/fit_smpl_sequence.py`)이 정확히 이 예외에 해당합니다. sequence별
성별/키를 알아야 SMPL의 shape(beta) 파라미터를 실측 키에 맞춰 보정할 수 있는데, 이는
곧 어떤 sequence들이 같은 피험자인지 드러내는 정보입니다.

## 정책

- sequence별 성별/키 매핑은 **이 저장소에 commit하지 않습니다**. `.gitignore`로도
  제외되며, private dataset root 바깥의 별도 파일(예: `subject_heights.json`)로만
  존재합니다.
- `configs/subject_anthropometry.example.json`은 스키마만 보여주는 템플릿이며 실제
  값은 없습니다 (`configs/paths.example.env`와 동일한 패턴).
- `tools/fit_smpl_sequence.py --height-cm`/`--gender`와 `tools/fit_smpl_all.py
  --subject-config`는 이 값을 CLI 인자/외부 파일로만 받고, 코드 안에 하드코딩하지
  않습니다.
- SMPL fit 출력(`outputs/smpl_fit_full/`)의 `metadata.json`에는 `height_cm_target`,
  `gender` 필드가 남습니다 — 이는 private output이며 (`outputs/`는 Git에서 제외),
  public 집계 CSV(`metadata/results/`)에는 포함하지 않습니다.
