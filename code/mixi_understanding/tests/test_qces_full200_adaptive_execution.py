from __future__ import annotations

import unittest

from mixi_understanding.qces.full200_adaptive_execution import (
    Full200ExecutionError,
    partition_joint_rows,
)


def row(video: str, crops: int, row_group: int | None = None) -> dict:
    output = {
        "video_id": video,
        "crop_requests": [
            {"selection_key": f"{video}-{index}"} for index in range(crops)
        ],
    }
    if row_group is not None:
        output["availability_location"] = {
            "parquet_url": "fixture.parquet",
            "row_group": row_group,
        }
    return output


class Full200AdaptiveExecutionTest(unittest.TestCase):
    def test_partition_respects_both_bounds_without_splitting_video(self) -> None:
        chunks = partition_joint_rows(
            [row("a", 2), row("b", 2), row("c", 1), row("d", 3)],
            maximum_videos=2,
            maximum_crops=3,
        )
        self.assertEqual([[value["video_id"] for value in chunk] for chunk in chunks], [["a"], ["b", "c"], ["d"]])
        self.assertEqual(sum(len(chunk) for chunk in chunks), 4)

    def test_single_large_video_remains_atomic(self) -> None:
        chunks = partition_joint_rows(
            [row("large", 5), row("small", 1)],
            maximum_videos=2,
            maximum_crops=3,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0]["video_id"], "large")

    def test_parquet_row_group_is_never_split_across_chunks(self) -> None:
        chunks = partition_joint_rows(
            [
                row("a", 1, 7),
                row("b", 1, 8),
                row("c", 1, 7),
                row("d", 1, 9),
            ],
            maximum_videos=2,
            maximum_crops=2,
        )
        containing = [
            index
            for index, chunk in enumerate(chunks)
            if any(value["video_id"] in {"a", "c"} for value in chunk)
        ]
        self.assertEqual(containing, [0])
        self.assertEqual(
            [value["video_id"] for value in chunks[0]], ["a", "c"]
        )

    def test_invalid_joint_row_fails(self) -> None:
        with self.assertRaises(Full200ExecutionError):
            partition_joint_rows(
                [{"video_id": "empty", "crop_requests": []}],
                maximum_videos=2,
                maximum_crops=3,
            )


if __name__ == "__main__":
    unittest.main()
