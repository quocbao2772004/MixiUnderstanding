from mixi_understanding.scripts.fetch_qces_agkphysics_direct_assets import (
    SUPPORTED_DATASET,
    SUPPORTED_REVISION,
    viewer_coordinate,
)


SHARDS = {
    ("unbal_train", 3): {
        "prefix": 6000,
        "num_rows": 2000,
        "row_group_prefix": {index: index * 100 for index in range(20)},
    },
    ("bal_train", 2): {
        "prefix": 1000,
        "num_rows": 500,
        "row_group_prefix": {index: index * 100 for index in range(5)},
    },
    ("eval", 11): {
        "prefix": 5500,
        "num_rows": 500,
        "row_group_prefix": {index: index * 100 for index in range(5)},
    },
}


def _row(path: str, row_group: int, row_index: int) -> dict:
    return {
        "video_id": "example",
        "availability_location": {
            "hf_dataset": SUPPORTED_DATASET,
            "hf_revision": SUPPORTED_REVISION,
            "parquet_url": (
                f"hf://datasets/agkphysics/AudioSet@{SUPPORTED_REVISION}/data/{path}"
            ),
            "row_group": row_group,
            "row_index": row_index,
        },
    }


def test_unbalanced_viewer_coordinate() -> None:
    assert viewer_coordinate(_row("unbal_train/003.parquet", 0, 14), SHARDS) == (
        "unbal_train",
        6014,
    )


def test_balanced_and_eval_viewer_coordinates() -> None:
    assert viewer_coordinate(_row("bal_train/02.parquet", 3, 7), SHARDS) == (
        "bal_train",
        1307,
    )
    assert viewer_coordinate(_row("eval/11.parquet", 4, 99), SHARDS) == ("eval", 5999)
