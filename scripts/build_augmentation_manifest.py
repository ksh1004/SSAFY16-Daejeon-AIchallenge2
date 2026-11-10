"""all_augmentation.ipynb 결과를 Qwen3.8 노트북용 증강 manifest로 변환한다.

notebooks/02_qwen38_27b/qwen38_27b_ocr_final.ipynb는 증강 CSV에
source_id / augmentation / is_augmented 열이 있어야 원본 이미지별로
증강 view를 고를 수 있다(Random View 샘플링). all_augmentation.ipynb가
만드는 train_all_augmented.csv에는 이 열이 없으므로 id 접미사를 해석해 추가한다.
Windows에서 생성된 역슬래시 경로도 Colab(Linux)용 '/'로 바꾼다.

사용법:
    python scripts/build_augmentation_manifest.py \
        --input  /content/train_all_augmented.csv \
        --output /content/train_all_augmented.csv
"""

import argparse
from pathlib import Path

import pandas as pd

AUGMENTATIONS = (
    "rot_p5", "rot_m5", "rot_p10", "rot_m10", "perspective",
    "downscale", "blur", "jpeg", "textcrop",
)


def describe(row_id: str):
    """id 접미사로 (원본 id, 증강 이름, 증강 여부)를 돌려준다."""
    for name in AUGMENTATIONS:
        suffix = "_" + name
        if row_id.endswith(suffix):
            return row_id[: -len(suffix)], name, True
    if row_id.startswith("dev_"):
        # Dev 3표 합의 pseudo-label. Qwen3.8 노트북은 train 원본의 증강만 사용하므로 제외된다.
        return row_id, "dev_pseudo_label", False
    return row_id, "original", False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="all_augmentation.ipynb의 train_all_augmented.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["id"] = frame["id"].astype(str)
    frame["path"] = frame["path"].astype(str).str.replace("\\", "/", regex=False)
    frame[["source_id", "augmentation", "is_augmented"]] = pd.DataFrame(
        frame["id"].map(describe).tolist(), index=frame.index
    )

    originals = set(frame.loc[frame["augmentation"] == "original", "id"])
    orphans = frame["is_augmented"] & ~frame["source_id"].isin(originals)
    if orphans.any():
        raise ValueError(f"원본 행이 없는 증강 행 {int(orphans.sum())}개: {frame.loc[orphans, 'id'].head().tolist()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"저장: {args.output} ({len(frame):,}행)")
    print(frame["augmentation"].value_counts().to_string())


if __name__ == "__main__":
    main()
