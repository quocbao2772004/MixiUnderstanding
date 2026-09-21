#!/usr/bin/env python3
"""Activity span refiner using two-event evidence for first questions."""

from __future__ import annotations

import sys

from mixi_understanding.scripts import calibrate_qces_v6_activity_span_refiner as base
from mixi_understanding.scripts.qces_v6_firstpair_candidate_utils import (
    prepare_named_split_firstpair,
)


if __name__ == "__main__":
    base.FORMAT_VERSION = "qces_v6_firstpair_activity_span_refiner_v1"
    base.prepare_named_split = prepare_named_split_firstpair
    base.main(sys.argv[1:])
