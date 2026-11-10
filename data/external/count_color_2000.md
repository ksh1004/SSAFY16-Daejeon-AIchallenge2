# 색상·개수 추가 학습 데이터

## 출처와 구성

원본은 [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)의 train 이미지이다. 로컬 파일명 `CLEVR_train_*.png`, 질문 ID, selection_manifest.csv가 근거다. 실사 사진이 아니라 색·모양·크기·재질이 다른 3D 물체를 배치한 합성 이미지로, 물체 속성과 관계를 이용한 시각 추론을 다룬다. 현재 묶음은 색상 및 개수 질문을 선별한 파생 데이터이며 원본 CLEVR 전체가 아니다.

## 실제 파일과 특성

- 이미지: `images/` 아래 PNG 2,000장, 모두 480×320 픽셀, RGBA.
- 대표 학습 질문: `train.csv` 2,000행. 색상 1,000문항, 개수 1,000문항.
- 질문·보기: 영어. 정답은 a/b/c/d이며 각 500개.
- 색상/개수 × 정답 위치의 8개 조합을 각 250개씩 선별했다.
- `clevr_count_color_qa.csv`, `clevr_count_color_train.csv`: 각각 2,131문항. 선택 이미지에 연결된 추가 질문을 보존한 파일이므로 train.csv와 단순 합산하지 않는다.
- 예시 질문: “How many small brown objects are the same shape as the big yellow thing?”처럼 단순 인식뿐 아니라 조건·속성·관계 해석이 필요하다.

## 선별 및 학습 참고

selection_report.json에 따르면 원본 후보 6,000문항/5,777개 경로에서 이미지 2,000장을 선택했으며 시드는 20260922이다. 당시 디코딩 검사 통과 및 동일 픽셀 중복 0건이 기록되어 있다. 이번 문서 작성에서 중복 검사를 다시 수행한 것은 아니다.

CSV 열은 `id,path,question,a,b,c,d,answer`이다. test.csv는 같은 2,000문항에서 answer만 제거한 추론용 파일로 독립 평가 세트가 아니다. 현재 CSV의 `count_color_selected_2000/images/...` 경로는 현재 폴더명과 다르므로 `count_color_2000/images/...`로 매핑해야 한다.

모델 입력 시 RGBA를 RGB로 일관되게 변환한다. 색상 증강은 정답을 바꿀 수 있고, 크롭은 개수나 공간 관계를 바꿀 수 있다. 실제 사진과 외형 차이가 있으므로 이 묶음의 성능을 실사 성능과 동일하게 해석하지 않는다.

## 로컬 근거

`selection_report.json`, `selection_manifest.csv`, `format_report.json`, `README.txt`, `train.csv`. `_original_csv/`는 이전 CSV와 문서 보관본이며 별도 학습 데이터가 아니다.
