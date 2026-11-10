# 데이터 준비

이 저장소에는 이미지·CSV 데이터가 들어 있지 않다. 대회 데이터는 Kaggle 대회 페이지에서 받고, 팀이 가공한 `final_data`는 용량(약 17GB)과 외부 데이터셋 배포 조건 때문에 제외했다. 이 폴더에는 데이터 구성 설명만 둔다.

## 1. 대회 원본 데이터: `ssafy-16-2-ai.zip`

[Kaggle 대회 페이지](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28)에서 받는다. 압축 비밀번호는 대회에서 제공된 값을 환경 변수 `VQA_ZIP_PASSWORD`로 넘긴다(노트북에는 비밀번호를 적지 않았다).

```text
ssafy-16-2-ai/
├── train.csv              # id, path, question, a, b, c, d, answer      (6,714행)
├── test.csv               # id, path, question, a, b, c, d              (6,714행)
├── dev.csv                # id, path, question, a~d, answer1~answer5    (2,683행)
├── sample_submission.csv  # id, answer
├── train/  test/  dev/    # 이미지
```

## 2. 팀 가공 데이터: `final_data.zip`

3-Stage 커리큘럼 노트북(`notebooks/03_curriculum`)은 아래 구조를 기대한다. 압축을 풀면 노트북이 `modify_50000` 폴더를 기준으로 루트를 자동 탐색한다.

```text
final_data/
├── modify_50000/                  # Stage 2: 원본 + 증강 + Dev pseudo-label (68,242행)
│   ├── train_all_augmented.csv    # notebooks/01_data/all_augmentation.ipynb 출력
│   ├── train/  dev/               # 원본 이미지 (CSV 경로가 train/..., dev/...)
│   ├── train_rotate_aug/          # 회전 ±5°·±10°, 원근 변환   33,570장
│   ├── train_quality_aug/         # 저해상도, 블러, JPEG 압축  20,142장
│   └── train_text_crop/           # EasyOCR 글자 영역 크롭      5,790장
├── english_3000/                  # Stage 3: ST-VQA 영어 장면 텍스트   3,000문항
├── count_color_2000/              # Stage 3: CLEVR 색상·개수           2,000문항
├── chinese_1000/                  # Stage 3: RCTW-17 중국어 간판        1,000문항
├── japanese_1000/                 # Stage 3: JaWildText 일본어         1,025문항 (이미지 745장)
└── non_english_3000/              # Stage 3: COCO-Text/ST-VQA          1,267문항
```

`modify_50000`은 원본 `train.csv`, `dev.csv`, 이미지 폴더를 둔 작업 폴더에서 `all_augmentation.ipynb`를 실행하면 만들어진다.

## 3. 외부 데이터 상세

Stage 3용 외부 데이터 5종의 출처, 라벨 생성 방식, 주의사항은 [`external/`](external) 문서에 정리했다. 요약하면 다음과 같다.

| 폴더 | 출처 | 겨냥한 약점 | 주의 |
|---|---|---|---|
| [`count_color_2000`](external/count_color_2000.md) | CLEVR (합성) | 개수·색상 | 실사 아님, 도메인 차이 |
| [`english_3000`](external/english_3000.md) | ST-VQA | 영어 장면 텍스트 | 오답 보기는 휴리스틱 생성 |
| [`chinese_1000`](external/chinese_1000.md) | ICDAR 2017 RCTW-17 | 한자 간판 | 정답이 자동 OCR 결과 |
| [`japanese_1000`](external/japanese_1000.md) | JaWildText | 일본어 텍스트 | 질문은 자동 번역, 원래 평가용 데이터 |
| [`non_english_3000`](external/non_english_3000.md) | COCO-Text · ST-VQA | 다국어 텍스트 | 1,901장은 문답 없음 |

> 최종 앙상블에 들어간 세 모델 중 외부 데이터(Stage 3)를 실제로 학습한 모델은 없다. Qwen3.5-27B는 Stage 3에서 optimizer update 2회(warmup 구간)만 진행된 상태로 쓰였다. 자세한 내용은 루트 README의 [최종 앙상블 구성](../README.md#최종-앙상블-구성)을 참고한다.
