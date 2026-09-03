"""Thin wrapper over facebookresearch/sam3's video predictor.

This is the ONLY module allowed to import sam3/torch, and both are imported lazily
inside functions/methods so the rest of the package (tracks.py, overlay.py,
manifest.py, annotate.py --no-overlay paths, tests) runs fine on a CPU-only login
node without sam3 or torch installed.

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


class Sam3Runner:
    def __init__(self, checkpoint: str, device: str = "cuda", score_thresh: float | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self.score_thresh = score_thresh
        self._predictor = None

    def _load(self):
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
        print(f"sam3_runner: checkpoint variant={variant} kwargs={kwargs}", flush=True)
        predictor = build_sam3_video_predictor(**kwargs)
        self._predictor = predictor
        return predictor

    def track(self, video_path: str, prompt: str, max_frames: int | None = None) -> list[dict]:
        """Run SAM3 text-prompted tracking over a video.

        Returns a list of per-frame dicts:
            {"frame": int, "ids": [int, ...], "probs": [float, ...],
             "boxes_xywh_norm": [[x, y, w, h], ...] (0-1 normalised),
             "masks": [np.ndarray(H, W) bool, ...]}
        """
        predictor = self._load()
        session_id = None
        try:
            start_resp = predictor.handle_request(
                request=dict(type="start_session", resource_path=video_path)
            )
            session_id = start_resp["session_id"] if isinstance(start_resp, dict) else start_resp

            prompt_kwargs = dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
            if self.score_thresh is not None:
                prompt_kwargs["output_prob_thresh"] = self.score_thresh
            predictor.handle_request(prompt_kwargs)

            # propagate_in_video is a streaming request: handle_stream_request yields
            # {"frame_index": i, "outputs": {...}} per frame (default direction "both"; the prompt is
            # on frame 0 so "forward" covers the whole clip).
            prop_kwargs = dict(type="propagate_in_video", session_id=session_id,
                               propagation_direction="forward", start_frame_index=0)
            if max_frames is not None:
                prop_kwargs["max_frame_num_to_track"] = max_frames
            if self.score_thresh is not None:
                prop_kwargs["output_prob_thresh"] = self.score_thresh
            prop_resp = predictor.handle_stream_request(prop_kwargs)

            results = []
            for frame_idx, out in self._iter_propagation(prop_resp):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                results.append(self._parse_frame_output(frame_idx, out))
            return results
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request(dict(type="close_session", session_id=session_id))
                except Exception:
                    pass

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
        if hasattr(x, "cpu"):  # torch tensor
            x = x.cpu().numpy()
        import numpy as np

        return np.asarray(x)

    @staticmethod
    def _to_list(x):
        if hasattr(x, "cpu"):  # torch tensor
            x = x.cpu().numpy()
        if hasattr(x, "tolist"):  # numpy array
            return x.tolist()
        return list(x)
