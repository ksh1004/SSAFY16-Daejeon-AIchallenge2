"""Top-3 가중 Soft-voting 앙상블 (최종 제출, Public LB 0.96127).

세 모델이 test 6,714문항에 대해 저장한 a/b/c/d 확률 CSV를 가중 평균해
최종 제출 파일을 다시 만든다. GPU 없이 수 초 안에 실행된다.
가중치와 계산 순서는 notebooks/04_ensemble/ensemble_01_softvote.ipynb의
`top3` 설정과 동일하다.

사용법:
    python scripts/ensemble_top3.py
    python scripts/ensemble_top3.py --prob-dir results/probabilities \
        --output results/submissions/submission_top3_reproduced.csv \
        --reference results/submissions/submission_final_top3.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = np.array(["a", "b", "c", "d"])
EXPECTED_ROW_COUNT = 6714

# 모델 이름: (확률 CSV 파일명, 가중치)
TOP3_MODELS = {
    "qwen38_27b": ("qwen38_27b_test_probabilities_03_final_other.csv", 0.40),
    "gemma4_31b": ("gemma_31b_test_probabilities_03_final_other.csv", 0.35),
    "qwen35_27b": ("qwen35_27b_test_probabilities_03_final_other.csv", 0.25),
}
REFERENCE_MODEL = "qwen38_27b"  # 제출 행 순서의 기준

# 모델마다 확률 열 이름이 다르다 (Qwen3.8 노트북은 a/b/c/d, 3-Stage 노트북은 p_a~p_d).
PROBABILITY_COLUMN_SCHEMES = [
    ["p_a", "p_b", "p_c", "p_d"],
    ["a", "b", "c", "d"],
]

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_probabilities(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    columns = next(
        (scheme for scheme in PROBABILITY_COLUMN_SCHEMES if set(scheme) <= set(frame.columns)),
        None,
    )
    if columns is None:
        raise ValueError(f"a/b/c/d 확률 열을 찾지 못했습니다: {path}")
    if len(frame) != EXPECTED_ROW_COUNT:
        raise ValueError(f"행 수가 {EXPECTED_ROW_COUNT}이 아닙니다: {path} ({len(frame)})")

    ids = frame["id"].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError(f"중복 id가 있습니다: {path}")

    probabilities = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"NaN 또는 무한대 확률이 있습니다: {path}")
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    return pd.DataFrame(probabilities, index=ids, columns=LABELS)


def build_ensemble(prob_dir: Path) -> pd.DataFrame:
    frames = {name: load_probabilities(prob_dir / file) for name, (file, _) in TOP3_MODELS.items()}

    reference_ids = frames[REFERENCE_MODEL].index
    for name, frame in frames.items():
        if set(frame.index) != set(reference_ids):
            raise ValueError(f"{name}의 id 구성이 {REFERENCE_MODEL}과 다릅니다.")

    total_weight = sum(weight for _, weight in TOP3_MODELS.values())
    combined = np.zeros((len(reference_ids), 4), dtype=np.float64)
    for name, (_, weight) in TOP3_MODELS.items():
        combined += (weight / total_weight) * frames[name].loc[reference_ids].to_numpy()
    combined /= combined.sum(axis=1, keepdims=True)

    return pd.DataFrame({"id": reference_ids, "answer": LABELS[combined.argmax(axis=1)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prob-dir", type=Path, default=REPO_ROOT / "results" / "probabilities")
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "results" / "submissions" / "submission_top3_reproduced.csv",
    )
    parser.add_argument(
        "--reference", type=Path,
        default=REPO_ROOT / "results" / "submissions" / "submission_final_top3.csv",
        help="비교할 기존 제출 파일. 없으면 비교를 건너뛴다.",
    )
    args = parser.parse_args()

    submission = build_ensemble(args.prob_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, lineterminator="\n")
    print(f"저장: {args.output} ({len(submission):,}행)")
    print("정답 분포:", submission["answer"].value_counts().sort_index().to_dict())

    if args.reference.is_file():
        reference = pd.read_csv(args.reference).set_index("id")["answer"]
        mismatches = int((reference.loc[submission["id"]].to_numpy() != submission["answer"].to_numpy()).sum())
        print(f"기존 제출과 다른 답: {mismatches}개 (기준: {args.reference.name})")


if __name__ == "__main__":
    main()
