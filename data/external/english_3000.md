# 영어 장면 텍스트 추가 학습 데이터

## 출처와 구성

직접 가져온 데이터는 [vikhyatk/st-vqa](https://huggingface.co/datasets/vikhyatk/st-vqa)의 train split이다. `st_vqa_images.csv`의 source_dataset, source_split, source_image_index가 이미지별 출처를 기록한다. 기반 과제는 [ST-VQA](https://arxiv.org/abs/1905.13648)이며, 사진 속 글자를 읽어 질문에 답하는 장면 텍스트 VQA이다. 이 저장소는 배포 경로이며 개별 사진 촬영자나 원본 이미지 URL은 현재 메타데이터에 없다.

## 실제 파일과 특성

- `images/`: JPG 3,000장. RGB 2,991장, 흑백 L 9장.
- 너비 168~4,000, 높이 187~2,592 픽셀. 두 범위의 최솟값·최댓값이 같은 이미지의 크기를 뜻하지는 않는다.
- 흔한 해상도: 500×375(392장), 640×480(273장), 500×333(222장), 1280×720(162장).
- `train.csv`, `selected_qa.csv`: 각각 대표 질문 3,000개. 영어 질문과 원문 문자열 보기.
- `st_vqa_english_qa.csv`: 선택 이미지에 연결된 전체 4,833문항.
- 정답 a/b/c/d는 각각 750개.
- 선별 유형: 브랜드 481, 이름 480, 숫자 480, 기타 358, 출판물 241, 표지·간판 480, 텍스트 480문항.
- 질문 기준으로 상품 브랜드, 책의 저자, 간판 문구, 숫자 등 글자와 주변 시각 문맥을 연결하는 학습에 해당한다.

## 가공 및 학습 참고

selection_report.json에 따르면 26,074문항/18,921개 이미지 경로에서 선별했다. 명시적인 텍스트 질문 여부, 부적절한 문답, 모호한 답변, 작은 이미지 등을 검사했고 시드는 20260922이다. 당시 동일 픽셀 중복 0건이 기록되어 있다.

원문 정답을 유지하고 다른 학습 답변에서 유형·형태가 비슷한 오답을 생성해 4지선다로 변환했다. 알려진 동일 이미지 정답 및 정규화·부분 문자열 동치 후보를 제외했지만, 모든 오답을 이미지와 대조해 수동 검수한 것은 아니다. `distractor_generation_audit.json`에 생성 기록이 있다.

CSV 열은 `id,path,question,a,b,c,d,answer`이다. test.csv는 train.csv의 정답 제거본이며 독립 평가 세트가 아니다. 경로 접두사 `english_selected_3000/`는 현재 `english_3000/`로 매핑해야 한다.

작은 글자가 사라지지 않도록 해상도를 관리하고 흑백 이미지는 RGB로 통일한다. 여러 질문이 한 이미지에 연결되므로 평가 분리는 이미지 기준으로 한다. non_english_3000도 ST-VQA 문답을 포함하므로 합칠 때 이미지 중복을 확인해야 한다. 폴더 사이 중복 수는 이번 문서에서 측정하지 않았다.

## 로컬 근거

`st_vqa_images.csv`, `selection_manifest.csv`, `selection_report.json`, `format_report.json`, `README.txt`, `distractor_generation_audit.json`. `_original_csv/`는 이전 버전 보관본이다.
