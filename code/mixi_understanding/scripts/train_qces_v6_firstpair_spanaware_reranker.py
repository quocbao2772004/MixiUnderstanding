#!/usr/bin/env python3
"""Train QCES-v6 span-aware reranker with two-event evidence for first questions."""

from __future__ import annotations

import sys

from mixi_understanding.scripts import train_qces_v6_spanaware_pair_reranker as base
from mixi_understanding.scripts.qces_v6_firstpair_candidate_utils import (
    split_candidates_spanaware_firstpair,
)


if __name__ == "__main__":
    base.FORMAT_VERSION = "qces_v6_firstpair_spanaware_pair_reranker_v1"
    base.split_candidates_spanaware = split_candidates_spanaware_firstpair
    base.main(sys.argv[1:])
