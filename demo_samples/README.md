# QCES demo samples

These small FLAC fixtures are curated from the local AudioQA smoke tests. They
are intended for a quick manual check of the public QCES demo; they are not a
training or benchmark dataset.

| Sample | What to try in the demo |
| --- | --- |
| [car_speech_mixture.flac](car_speech_mixture.flac) | Ask which vehicle/road sounds are present, then ask what the speaker said. |
| [overlap_mixture.flac](overlap_mixture.flac) | Two-source overlap mixture for the MossFormer2 separation path. |
| [speaker_1_overlap_reference.flac](speaker_1_overlap_reference.flac) | Reference source 1 from the overlap smoke test. |
| [speaker_2_overlap_reference.flac](speaker_2_overlap_reference.flac) | Reference source 2 from the overlap smoke test. |

All files are mono FLAC at 16 kHz. The live detector is intentionally capped at
10 seconds; the car sample is kept at its original 26.1-second length so it can
also be used to verify the long-input warning and first-window inference.
