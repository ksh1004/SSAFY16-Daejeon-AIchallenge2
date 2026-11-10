# 일본어 장면 텍스트 추가 학습 데이터

## 출처와 구성

원본은 [llm-jp/JaWildText](https://huggingface.co/datasets/llm-jp/jawildtext)의 `board_vqa` 구성이다. 엑셀 source 열과 설명 시트에서 확인된다. 일본어 Dense Scene Text VQA로, 게시물·상품 설명 등 밀집된 글자를 읽고 여러 텍스트 근거를 연결하는 과제다. 원본 소개는 [JaWildText 논문](https://arxiv.org/abs/2603.27942)에서 확인할 수 있다.

## 실제 파일과 특성

- **폴더명은 japanese_1000이지만 실제 이미지는 745장**, 문항은 1,025개다.
- 폴더 바로 아래 JPG 745장, 모두 RGB.
- 너비 1,034~5,712, 높이 945~5,712 픽셀. 너비·높이 범위는 각각 집계했다.
- 흔한 해상도: 4000×3000(88장), 4032×3024(71장), 3024×4032(69장), 1536×2048(65장).
- `일본어 vqa.xlsx`: 질문 1,025행과 이미지 목록 745행.
- question_ja는 일본어 원문, question_ko는 자동 번역한 한국어 질문이다.
- 보기와 정답은 일본어 원문을 유지한다.
- 정답 위치: A 248, B 256, C 257, D 264.
- 문항 예시는 상품 사용 횟수, 경기 결과에 따른 조건, 문서 내 수치 계산 등이다.

## 가공 및 학습 참고

엑셀 기록상 공식 board_vqa 문답을 4지선다로 변환했다. 같은 이미지의 다른 정답을 우선하고 전체 답변 풀에서 문자 유형·길이가 비슷한 오답을 보완했다. 한국어 자동 번역의 의미 보존과 오답의 유일성을 검수할 필요가 있다.

핵심 열: `question_id,image_id,file_name,image_path,question_ko,choice_A~D,correct_option,correct_answer,question_ja,evidence_polygon_ids,source,source_row_idx`.

evidence_polygon_ids는 원본 주석의 텍스트 근거 영역 ID다. 엑셀에 ID만 있으므로 실제 폴리곤 좌표가 필요하면 원본 주석과 연결해야 한다. image_path는 이전 위치를 가리키므로 현재 `japanese_1000/{file_name}`로 연결한다.

공식 데이터 카드는 Hugging Face의 split 이름이 train이어도 **원래 평가 벤치마크로 의도된 데이터**라고 설명한다. 추가 학습에 사용할 경우 동일 JaWildText 문항을 독립적인 미학습 평가에 재사용하지 않는다. 작은 글자가 많은 고해상도 이미지이므로 과도한 축소가 근거를 지울 수 있다.

## 로컬 근거

`일본어 vqa.xlsx`의 문항·이미지 목록·설명 시트. 이미지 수와 모드·해상도는 현재 파일을 직접 집계했다.
