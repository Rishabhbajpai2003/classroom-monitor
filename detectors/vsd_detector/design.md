# Visual Speech Detector Design

This design uses the current InsightFace-based face detector and tracker as the upstream region-of-interest backbone, then applies a VTP-style visual speech model inspired by:

- `Sub-word Level Lip Reading With Visual Attention` (Prajwal et al., 2021)
- `https://github.com/prajwalkr/vtp`

## What we kept from the paper

- A spatio-temporal visual front end instead of framewise classification.
- Visual Transformer Pooling (VTP): attention over each frame's spatial feature map rather than plain global average pooling.
- A temporal Transformer encoder over frame embeddings.
- A visual speech detector head on top of encoder outputs for frame-level speech probability.
- A lip-reading decoder that predicts sub-word tokens, not characters.

## What we adapted for this repo

- The existing face detector is used to produce stable face tracks. This is the practical "backbone" for classroom videos.
- We crop an expanded face region, not only lips. This matches the paper's observation that extra-oral regions can help and avoids relying on facial landmarks.
- The implementation is lighter than the Oxford training code so it fits the current repo and dependency set.
- The lip reader is intentionally decoder-first scaffolding: it is ready for checkpoint loading and integration, but it still needs training on aligned visual speech data.

## File layout

- `detectors/vsd_detector/common.py`
  Shared crop utilities, track clip buffering, VTP-style encoder, VSD head, and lip-reading model.
- `detectors/vsd_detector/run.py`
  Frame-level Visual Speech Detection over face tracks.
- `detectors/vsd_detector/train.py`
  Paper-aligned VSD training script that fine-tunes the official `silencer_vtp24x24` model on AVA-style frame labels.
- `detectors/vsd_detector/dataset.py`
  AVA ActiveSpeaker dataset loader that crops face tracks from raw videos and produces framewise speech labels.
- `detectors/vlp_detector/run.py`
  Lip reading over detected speaking segments, optionally gated by the VSD checkpoint.

## Runtime pipeline

1. Detect faces with `InsightFaceBackend`.
2. Track faces with the persistent `FaceTracker`.
3. For each track, extract an expanded face crop resized to `96x96`.
4. Feed short track clips into the VSD model.
5. Use VSD probabilities to decide when a face track is actively speaking.
6. Accumulate speaking segments and decode them with the lip-reading model.

## Model boundaries

- VSD can be trained first and used on its own.
- Lip reading should normally sit behind VSD; otherwise segmentation quality will be poor.
- The current code supports checkpoint loading, but does not claim pretrained weights exist in this repo.

## Training recommendations

- Start with VSD first. It is the easier target and gives immediate utility.
- Keep `96x96` face crops and `25 fps` where possible to stay close to the paper setup.
- Use the same visual encoder for both tasks so VSD and lip reading share the representation.
- Use WordPiece tokenization for lip reading, consistent with the paper.
- Fine-tune the encoder for classroom footage after initial training on public visual speech data if available.

## Practical next steps

1. Download AVA ActiveSpeaker train/val annotations and the corresponding AVA videos.
2. Fine-tune `silencer_vtp24x24` from the released lip-reading checkpoint using `train.py`.
3. Point `detectors/vsd_detector/run.py` at the resulting `best.pth`.
4. Train lip reading on aligned transcripts only after the VSD segmentation path is stable.
