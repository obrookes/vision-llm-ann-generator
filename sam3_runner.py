"""Thin wrapper over facebookresearch/sam3's video predictor.

This is the ONLY module allowed to import sam3/torch, and both are imported lazily
inside functions/methods so the rest of the package (tracks.py, overlay.py,
manifest.py, annotate.py --no-overlay paths, tests) runs fine on a CPU-only login
node without sam3 or torch installed.

SAM3 official API (facebookresearch/sam3, sam3/model_builder.py +
sam3/model/sam3_video_inference.py):
    from sam3.model_builder import build_sam3_video_predictor
    predictor.handle_request(request=dict(type="start_session", resource_path=video_path))
        -> response["session_id"]
    predictor.handle_request(dict(type="add_prompt", session_id=..., frame_index=0, text=prompt))
    predictor.handle_request(dict(type="propagate_in_video", session_id=...))
        -> per-frame outputs with out_obj_ids, out_probs, out_boxes_xywh (normalised 0-1),
           out_binary_masks (N,H,W bool)
    predictor.handle_request(dict(type="close_session", session_id=...))

Confirmed from source (sam3/model_builder.py): build_sam3_video_predictor(*args,
gpus_to_use=None, **kwargs) forwards kwargs to build_sam3_video_model(checkpoint_path:
str|None=None, load_from_HF=True, bpe_path=None, device="cuda", compile=False, ...).
We call it with checkpoint_path=<local .pt> and load_from_HF=False so it loads our
local SA-FARI weights instead of pulling from the Hub. The BPE tokenizer is bundled in
the package (sam3/assets/bpe_simple_vocab_16e6.txt.gz), so no HF/network download is
needed for text prompts.

Whether propagate_in_video returns a generator of (frame_idx, out) tuples vs. a
dict/list is still unconfirmed -- this wrapper handles both defensively; see
CONFIRM ON FIRST GPU RUN below.
"""
from __future__ import annotations


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

        # Confirmed API: build_sam3_video_predictor(**kwargs) forwards to
        # build_sam3_video_model(checkpoint_path=..., load_from_HF=False, device=...).
        # load_from_HF=False so it loads our local SA-FARI .pt instead of the Hub default.
        try:
            predictor = build_sam3_video_predictor(
                checkpoint_path=self.checkpoint, load_from_HF=False, device=self.device
            )
        except TypeError:
            # CONFIRM ON FIRST GPU RUN: fall back in case the kwarg name differs across
            # sam3 versions.
            predictor = build_sam3_video_predictor(ckpt_path=self.checkpoint, device=self.device)
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

            predictor.handle_request(
                dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
            )

            prop_resp = predictor.handle_request(
                dict(type="propagate_in_video", session_id=session_id)
            )

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

        CONFIRM ON FIRST GPU RUN: whether this is a generator of (frame_idx, out) tuples,
        a dict keyed by frame_idx, or a list of per-frame dicts carrying their own index.
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
