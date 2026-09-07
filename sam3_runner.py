"""Thin wrapper over facebookresearch/sam3's video predictor and image detector.

This is the ONLY module allowed to import sam3/torch, and both are imported lazily
inside functions/methods so the rest of the package (frames.py, tracks.py, overlay.py,
manifest.py, annotate.py --no-overlay paths, tests) runs fine on a CPU-only login
node without sam3 or torch installed.

Two inference modes, both operating on the SAME subsampled frame set (frames.py's
decode_frames): SAM 3 is trained/evaluated at 6 fps (see the SAM 3 / SA-FARI papers),
while our corpus is 24-30 fps, so by default only every 4th-5th source frame is decoded
and handed to the model (`sample_fps=6.0`; `0`/`None` disables subsampling and decodes
every source frame).

  - mode="video" (default): the sampled PIL frames are handed to sam3's video predictor
    as a list (`resource_path=[PIL, ...]`, confirmed accepted by
    sam3/model/io_utils.py:44-73); the prompt is added on the first (positional) sampled
    frame and propagated forward with SAM2-style memory tracking, giving persistent
    object ids across frames. Edge case: sam3/model/sam3_video_inference.py:1713-1716
    treats a *list of length 1* as a still image (`is_image_type`), which would disable
    propagation, so when only one frame survives subsampling we pass the source video
    path instead and let sam3 decode it itself (frame 0 is always kept by decode_frames,
    so this degrades to "prompt + read frame 0", matching the sampled set of one frame).
  - mode="image": each sampled frame is run independently through sam3's image detector
    (`build_sam3_image_model` + `sam3.model.sam3_image_processor.Sam3Processor`) - no
    cross-frame memory/tracking, so `ids` are just a per-frame 0..n-1 index. Boxes come
    back xyxy in ORIGINAL pixels and are converted to normalised xywh here to match video
    mode's output shape.

SAM3 API as installed (sam3 0.1.0 from facebookresearch/sam3, checked inside the container):
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(checkpoint_path=<local .pt>)   # HF download only if None
    predictor.handle_request(dict(type="start_session", resource_path=video)) -> {"session_id"}
    predictor.handle_request(dict(type="add_prompt", session_id, frame_index=0, text=prompt,
                                  output_prob_thresh=0.5))
    predictor.handle_stream_request(dict(type="propagate_in_video", session_id,
                                         propagation_direction="forward", ...))
        -> yields {"frame_index": i, "outputs": {out_obj_ids, out_probs,
                   out_boxes_xywh (normalised), out_binary_masks (N,H,W bool)}}
    predictor.handle_request(dict(type="close_session", session_id))

    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(checkpoint_path=<local .pt>, device=..., load_from_HF=False,
                                    enable_inst_interactivity=False)
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model)
    processor.set_confidence_threshold(0.5)
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(prompt=..., state=state)
    state["boxes"]   # xyxy, ORIGINAL pixels, filtered by confidence_threshold
    state["masks"]   # bool, (N, H, W) or (N, 1, H, W)
    state["scores"]

The BPE tokenizer is bundled in the package (sam3/assets/bpe_simple_vocab_16e6.txt.gz).

Propagation goes through handle_stream_request (a generator of
{"frame_index", "outputs"} dicts); _iter_propagation still tolerates the other shapes.
"""
from __future__ import annotations


def checkpoint_variant(checkpoint: str) -> str:
    """Peek at the state-dict keys (mmap, no GPU) and classify the presence mechanism.

    "decoder_presence_token": Meta's release sam3.pt (detector.transformer.decoder.presence_token.*)
    "seg_head_presence":      SA-FARI fine-tunes (detector.segmentation_head.presence_head.*)
    "unknown":                neither (loaded with the builder defaults)
    """
    import torch

    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    keys = list(sd.keys())
    if any(k.startswith("detector.segmentation_head.presence_head.") for k in keys):
        return "seg_head_presence"
    if any(k.startswith("detector.transformer.decoder.presence_token") for k in keys):
        return "decoder_presence_token"
    return "unknown"


def _patch_segmentation_head_with_presence():
    """Make the model match the SA-FARI checkpoints: a segmentation head with a DotProductScoring
    presence head (keys presence_head.{prompt_mlp,prompt_proj,hs_proj}) and a decoder without the
    presence token. Idempotent. Note the video inference path never
    reads the head's presence_logit, so this only exists to satisfy strict state-dict loading."""
    from sam3 import model_builder as mb

    if getattr(mb, "_presence_head_patched", False):
        return
    orig = mb._create_segmentation_head

    def patched(*args, **kwargs):
        head = orig(*args, **kwargs)
        head.presence_head = mb._create_dot_product_scoring()
        return head

    mb._create_segmentation_head = patched

    # sam3 0.1.0's _create_sam3_transformer(has_presence_token=...) ignores the flag and
    # _create_transformer_decoder hard-codes presence_token=True, so strip the decoder presence
    # token here as well (the decoder forward checks `self.presence_token is not None`).
    orig_dec = mb._create_transformer_decoder

    def patched_dec(*args, **kwargs):
        dec = orig_dec(*args, **kwargs)
        dec.presence_token = None
        dec.presence_token_head = None
        dec.presence_token_out_norm = None
        return dec

    mb._create_transformer_decoder = patched_dec
    mb._presence_head_patched = True


def _patch_image_processor_scoring():
    """Make sam3.model.sam3_image_processor.Sam3Processor._forward_grounding robust to a decoder
    with no presence token (our SA-FARI patch above sets decoder.presence_token = None).

    Verified inside the container:
      - sam3/model/sam3_image.py:340-343 (_update_scores_and_boxes) only writes
        out["presence_logit_dec"] when `dec_presence_out is not None`:
            if dec_presence_out is not None:
                _update_out(out, "presence_logit_dec", dec_presence_out, update_aux=self.training)
        `dec_presence_out` comes straight from `self.transformer.decoder(...)`, whose forward
        returns None for it when `self.presence_token is None` - exactly our patched state.
      - sam3/model/sam3_image_processor.py:196-197 (_forward_grounding) reads that key
        UNCONDITIONALLY:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence_score).squeeze(-1)
        -> with the decoder patch applied this is a bare KeyError, not a None-multiply.

    So this patches _forward_grounding to fall back to plain `sigmoid(pred_logits)` (no presence
    multiplier) whenever "presence_logit_dec" is absent from the model's output dict, and prints
    which scoring path is active (once) so a run's logs say which formula produced its scores.
    The SA-FARI segmentation head's own DotProductScoring presence_head (added by
    _patch_segmentation_head_with_presence) is not read by forward_grounding at all - it's a
    detection-side head for a different code path - so there's no drop-in "richer" presence signal
    to reach for here; plain sigmoid(pred_logits) is genuinely the best available score.
    """
    from sam3.model import sam3_image_processor as sip

    if getattr(sip.Sam3Processor, "_scoring_patched", False):
        return

    import torch
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate

    @torch.inference_mode()
    def patched(self, state):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence_score).squeeze(-1)
            scoring_path = "sigmoid(pred_logits) * sigmoid(presence_logit_dec)"
        else:
            out_probs = out_probs.squeeze(-1)
            scoring_path = (
                "sigmoid(pred_logits) only (no presence_logit_dec in model output; "
                "decoder presence token is patched to None for SA-FARI checkpoints)"
            )
        if not getattr(sip.Sam3Processor, "_scoring_path_printed", False):
            print(f"sam3_runner: [image mode] scoring path = {scoring_path}", flush=True)
            sip.Sam3Processor._scoring_path_printed = True

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    sip.Sam3Processor._forward_grounding = patched
    sip.Sam3Processor._scoring_patched = True


class Sam3Runner:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        score_thresh: float | None = None,
        mode: str = "video",
        sample_fps: float | None = 6.0,
    ):
        if mode not in ("video", "image"):
            raise ValueError(f"mode must be 'video' or 'image', got {mode!r}")
        self.checkpoint = checkpoint
        self.device = device
        self.score_thresh = score_thresh
        self.mode = mode
        self.sample_fps = sample_fps
        self._predictor = None
        self._image_model = None
        self._image_processor = None

    def _load(self):
        """Lazily build the video predictor (mode="video" only)."""
        if self._predictor is not None:
            return self._predictor
        from sam3.model_builder import build_sam3_video_predictor

        # Confirmed against the installed package (sam3 0.1.0, 2026-09-03): build_sam3_video_predictor
        # returns Sam3VideoPredictorMultiGPU(*args, gpus_to_use=None, **kwargs) whose per-GPU
        # Sam3VideoPredictor.__init__(checkpoint_path=None, bpe_path=None, has_presence_token=True,
        # ..., compile=False) calls build_sam3_video_model(checkpoint_path=...) and .cuda()s the model.
        # The HF download only happens when checkpoint_path is None.
        variant = checkpoint_variant(self.checkpoint)
        kwargs = dict(checkpoint_path=self.checkpoint)
        if variant == "seg_head_presence":
            # SA-FARI fine-tunes (sam3-safari-*.pt): no decoder presence token, but a
            # DotProductScoring presence head on the segmentation head. The builder has a flag for
            # the former and hard-codes presence_head=False for the latter, so patch the head factory.
            kwargs["has_presence_token"] = False
            _patch_segmentation_head_with_presence()
        print(f"sam3_runner: [video mode] checkpoint variant={variant} kwargs={kwargs}", flush=True)
        predictor = build_sam3_video_predictor(**kwargs)
        self._predictor = predictor
        return predictor

    def _load_image_model(self):
        """Lazily build the image detector + processor (mode="image" only)."""
        if self._image_model is not None:
            return self._image_model, self._image_processor
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        variant = checkpoint_variant(self.checkpoint)
        # model_builder.py:573-582 signature (checked inside the container): bpe_path, device,
        # eval_mode, checkpoint_path, load_from_HF, enable_segmentation, enable_inst_interactivity,
        # compile. load_from_HF only matters when checkpoint_path is None, but pass it explicitly
        # (False) so a bad local path can never silently fall back to a HF download.
        kwargs = dict(
            checkpoint_path=self.checkpoint,
            device=self.device,
            load_from_HF=False,
            enable_inst_interactivity=False,
        )
        if variant == "seg_head_presence":
            # Same reasoning as _load(): SA-FARI checkpoints need the segmentation-head presence
            # patch applied BEFORE the model is built, so the state dict loads into a matching
            # module set (model_builder._load_checkpoint loads strict=False and prints
            # missing_keys itself, see model_builder.py:539-561).
            _patch_segmentation_head_with_presence()
        _patch_image_processor_scoring()
        print(f"sam3_runner: [image mode] checkpoint variant={variant} kwargs={kwargs}", flush=True)
        model = build_sam3_image_model(**kwargs)
        processor = Sam3Processor(model, device=self.device)
        if self.score_thresh is not None:
            processor.set_confidence_threshold(self.score_thresh)
        self._image_model = model
        self._image_processor = processor
        return model, processor

    def track(
        self,
        video_path: str,
        prompt: str,
        max_frames: int | None = None,
        extra_indices=None,
        stop_after: int | None = None,
    ) -> tuple[list[dict], dict]:
        """Run SAM3 text-prompted detection/tracking over a (fps-subsampled) video.

        `extra_indices` and `stop_after` are passed straight through to
        `frames.decode_frames` (force-keep specific source frame indices even if they're off
        the sampling grid; stop decoding after a given source index). SAM3 itself only ever
        sees the resulting sampled frame list/positional indices, so nothing downstream of
        decode_frames needs to know about them - the existing `frame_indices` mapping already
        carries the (possibly irregular) source indices back onto SAM3's positional output.

        Returns (frames, meta):
            frames: list of per-frame dicts, `frame` = SOURCE video frame index:
                {"frame": int, "ids": [int, ...], "probs": [float, ...],
                 "boxes_xywh_norm": [[x, y, w, h], ...] (0-1 normalised),
                 "masks": [np.ndarray(H, W) bool, ...]}
            meta: {"src_fps": float, "n_frames_sampled": int,
                   "sample_fps": float | None, "mode": "video"|"image"}
        """
        from frames import decode_frames

        frame_indices, pil_frames, src_fps = decode_frames(
            video_path, self.sample_fps, max_frames=max_frames,
            extra_indices=extra_indices, stop_after=stop_after,
        )
        meta = {
            "src_fps": src_fps,
            "n_frames_sampled": len(frame_indices),
            "sample_fps": self.sample_fps if self.sample_fps else None,
            "mode": self.mode,
        }
        if not frame_indices:
            return [], meta

        if self.mode == "image":
            results = self._track_image(pil_frames, frame_indices, prompt)
        else:
            results = self._track_video(pil_frames, frame_indices, prompt, video_path)
        return results, meta

    def _track_video(self, pil_frames: list, frame_indices: list[int], prompt: str, video_path: str) -> list[dict]:
        predictor = self._load()
        session_id = None
        # sam3_video_inference.py:1713-1716 (is_image_type) treats a list of length 1 as a still
        # image, which would disable propagation, so fall back to the source video path in that
        # case and let sam3 decode it itself.
        use_pil_list = len(pil_frames) > 1
        resource = pil_frames if use_pil_list else video_path
        try:
            start_resp = predictor.handle_request(request=dict(type="start_session", resource_path=resource))
            session_id = start_resp["session_id"] if isinstance(start_resp, dict) else start_resp

            prompt_kwargs = dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
            if self.score_thresh is not None:
                prompt_kwargs["output_prob_thresh"] = self.score_thresh
            predictor.handle_request(prompt_kwargs)

            # propagate_in_video is a streaming request: handle_stream_request yields
            # {"frame_index": i, "outputs": {...}} per frame (default direction "both"; the prompt is
            # on frame 0 so "forward" covers the whole clip). `frame_index` here is POSITIONAL into
            # whatever resource_path we handed sam3 (the sampled frame list, or the full source
            # video in the len==1 fallback) - map it back to a source-video index via frame_indices.
            prop_kwargs = dict(type="propagate_in_video", session_id=session_id,
                               propagation_direction="forward", start_frame_index=0)
            if self.score_thresh is not None:
                prop_kwargs["output_prob_thresh"] = self.score_thresh
            prop_resp = predictor.handle_stream_request(prop_kwargs)

            results = []
            for i, (frame_index, out) in enumerate(self._iter_propagation(prop_resp)):
                if i >= len(frame_indices):
                    break
                # use_pil_list: frame_index is positional into the sampled list -> map through
                # frame_indices. Fallback (full video decode): only frame_indices[0] (== 0, since
                # decode_frames always keeps frame 0) is wanted, i.e. just the first yielded frame.
                source_idx = frame_indices[frame_index] if use_pil_list else frame_indices[i]
                results.append(self._parse_frame_output(source_idx, out))
            return results
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request(dict(type="close_session", session_id=session_id))
                except Exception:
                    pass

    def _track_image(self, pil_frames: list, frame_indices: list[int], prompt: str) -> list[dict]:
        model, processor = self._load_image_model()

        import torch

        results = []
        for source_idx, pil in zip(frame_indices, pil_frames):
            # The video path runs under @torch.autocast(bfloat16) (sam3_video_inference.py:800,908);
            # the image processor does not, and the ViT trunk mixes bf16 activations with fp32
            # weights without it ("mat1 and mat2 must have the same dtype"). Match the video path.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil)
                state = processor.set_text_prompt(prompt=prompt, state=state)

            boxes_xyxy = self._to_numpy(state["boxes"])  # xyxy, ORIGINAL pixels
            masks = self._to_numpy(state["masks"])  # bool, (N, H, W) or (N, 1, H, W)
            scores = self._to_list(state["scores"])

            width, height = pil.size
            n = boxes_xyxy.shape[0]
            boxes_norm = []
            for x1, y1, x2, y2 in boxes_xyxy:
                boxes_norm.append([
                    float(x1) / width,
                    float(y1) / height,
                    float(x2 - x1) / width,
                    float(y2 - y1) / height,
                ])

            if masks.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
                masks = masks[:, 0]
            masks_list = [masks[i].astype(bool) for i in range(masks.shape[0])]

            results.append({
                "frame": int(source_idx),
                "ids": list(range(n)),  # no cross-frame tracking in image mode
                "probs": [float(p) for p in scores],
                "boxes_xywh_norm": boxes_norm,
                "masks": masks_list,
            })
        return results

    @staticmethod
    def _iter_propagation(prop_resp):
        """Normalise propagate_in_video's response into an iterable of (frame_idx, out).

        The installed sam3 yields {"frame_index", "outputs"} dicts; the other shapes are kept
        for robustness against future versions.
        """
        # Generator / iterable of (frame_idx, out) tuples.
        if hasattr(prop_resp, "__iter__") and not isinstance(prop_resp, dict):
            for item in prop_resp:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    yield item[0], item[1]
                elif isinstance(item, dict) and "frame_index" in item:
                    # handle_request-style streaming: {"frame_index": i, "outputs": {...}}
                    yield item["frame_index"], item.get("outputs", item)
                else:
                    raise TypeError(f"unrecognised propagate_in_video item shape: {type(item)}")
            return
        # Dict keyed by frame index.
        if isinstance(prop_resp, dict):
            if "results" in prop_resp:
                prop_resp = prop_resp["results"]
            if isinstance(prop_resp, dict):
                for k, v in prop_resp.items():
                    yield int(k), v
                return
            if isinstance(prop_resp, list):
                for i, v in enumerate(prop_resp):
                    yield v.get("frame_index", i) if isinstance(v, dict) else i, v
                return
        raise TypeError(f"unrecognised propagate_in_video response shape: {type(prop_resp)}")

    def _parse_frame_output(self, frame_idx: int, out) -> dict:
        obj_ids = self._to_list(out["out_obj_ids"])
        probs = self._to_list(out["out_probs"])
        boxes = self._to_list(out["out_boxes_xywh"])
        masks_raw = out.get("out_binary_masks")
        masks = []
        if masks_raw is not None:
            masks_np = self._to_numpy(masks_raw)
            masks = [masks_np[i].astype(bool) for i in range(masks_np.shape[0])]

        if self.score_thresh is not None and probs:
            keep = [i for i, p in enumerate(probs) if p >= self.score_thresh]
            obj_ids = [obj_ids[i] for i in keep]
            probs = [probs[i] for i in keep]
            boxes = [boxes[i] for i in keep]
            masks = [masks[i] for i in keep] if masks else masks

        return {
            "frame": int(frame_idx),
            "ids": [int(i) for i in obj_ids],
            "probs": [float(p) for p in probs],
            "boxes_xywh_norm": [[float(v) for v in b] for b in boxes],
            "masks": masks,
        }

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "cpu"):  # torch tensor (may be bf16 under autocast; numpy has no bf16)
            if hasattr(x, "is_floating_point") and x.is_floating_point():
                x = x.float()
            x = x.cpu().numpy()
        import numpy as np

        return np.asarray(x)

    @staticmethod
    def _to_list(x):
        if hasattr(x, "cpu"):  # torch tensor (may be bf16 under autocast; numpy has no bf16)
            if hasattr(x, "is_floating_point") and x.is_floating_point():
                x = x.float()
            x = x.cpu().numpy()
        if hasattr(x, "tolist"):  # numpy array
            return x.tolist()
        return list(x)
