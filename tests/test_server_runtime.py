import contextlib
import io
import sys
import types
import unittest
from importlib import import_module

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import BBox, TrackingResult
from iot_servo_tracker.server.main import process_frame_safely
from iot_servo_tracker.server.runtime import VisionRuntime
from iot_servo_tracker.server.vision import HuggingFaceWeDetectRefClient, WeDetectYoloPipeline


class TensorList:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class TensorItems:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return TensorScalar(self.data[index])


class TensorScalar:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def item(self):
        return self.value


class FakeBoxes:
    def __init__(self, candidates):
        self.xyxy = TensorList([candidate[0] for candidate in candidates])
        self.conf = TensorItems([candidate[1] for candidate in candidates])
        track_ids = [candidate[2] for candidate in candidates]
        self.id = None if all(track_id is None for track_id in track_ids) else TensorItems(track_ids)
        class_ids = [candidate[3] if len(candidate) > 3 else None for candidate in candidates]
        self.cls = None if all(class_id is None for class_id in class_ids) else TensorList(class_ids)


class FakeResult:
    def __init__(self, candidates, names=None):
        self.boxes = FakeBoxes(candidates)
        self.names = names


class FakeYolo:
    def __init__(self, results, names=None):
        self.results = results
        self.names = names

    def track(self, *args, **kwargs):
        return self.results


class EmptyRedetectClient:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        self.calls += 1
        return TrackingResult.empty(ts_req=ts_req, query=query)


def build_test_pipeline(yolo, wedetect_client=None):
    pipeline = object.__new__(WeDetectYoloPipeline)
    pipeline.yolo = yolo
    pipeline.tracker = "bytetrack.yaml"
    pipeline.confidence_threshold = 0.25
    pipeline.yolo_lost_frames = 2
    pipeline.yolo_suspect_frames = 1
    pipeline.yolo_max_center_jump_px = 120.0
    pipeline.yolo_max_area_growth_ratio = 4.0
    pipeline.yolo_max_frame_area_ratio = 0.35
    pipeline.yolo_max_aspect_ratio_change = 3.0
    pipeline.yolo_min_iou_on_id_change = 0.10
    pipeline.camera = AppConfig().camera
    pipeline.wedetect_client = wedetect_client or EmptyRedetectClient()
    pipeline.locked_bbox = BBox(300, 220, 340, 260)
    pipeline.locked_track_id = 7
    pipeline.redetect_reference_bbox = None
    pipeline.yolo_lost_count = 0
    pipeline.yolo_suspect_count = 0
    pipeline.last_yolo_reject_reason = ""
    pipeline._decode_frame = lambda frame_bytes: object()
    return pipeline


class RecordingPipeline:
    def __init__(self) -> None:
        self.frame_bytes = b""
        self.redetect_count = 0

    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
    ) -> TrackingResult:
        self.frame_bytes = frame_bytes
        return TrackingResult.empty(ts_req=ts_req, query=query)

    def redetect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        self.redetect_count += 1
        self.frame_bytes = frame_bytes
        return TrackingResult.empty(ts_req=ts_req, query=query)


class VisionRuntimeTests(unittest.TestCase):
    def test_passes_frame_bytes_to_pipeline(self) -> None:
        pipeline = RecordingPipeline()
        runtime = VisionRuntime(AppConfig(), pipeline=pipeline)
        runtime.process(ts_req=1, query="red cup", frame_bytes=b"jpeg-bytes")
        self.assertEqual(pipeline.frame_bytes, b"jpeg-bytes")

    def test_redetect_flag_calls_pipeline_redetect(self) -> None:
        pipeline = RecordingPipeline()
        runtime = VisionRuntime(AppConfig(), pipeline=pipeline)
        runtime.process(ts_req=1, query="red cup", frame_bytes=b"jpeg-bytes", redetect=True)
        self.assertEqual(pipeline.redetect_count, 1)
        self.assertEqual(pipeline.frame_bytes, b"jpeg-bytes")

    def test_huggingface_wedetect_ref_module_adapter_returns_tracking_result(self) -> None:
        module = types.ModuleType("fake_wedetect_ref")

        def detect(**kwargs):
            self.assertEqual(kwargs["frame_bytes"], b"jpeg-bytes")
            self.assertEqual(kwargs["query"], "red cup")
            self.assertEqual(kwargs["wedetect_ref_model_dir"], "ref-dir")
            self.assertEqual(kwargs["wedetect_uni_checkpoint"], "uni.pth")
            self.assertEqual(kwargs["device"], "cpu")
            return {"bbox": [10, 20, 50, 80], "confidence": 0.91, "track_id": 42}

        module.detect = detect
        sys.modules[module.__name__] = module
        try:
            client = HuggingFaceWeDetectRefClient(
                ref_model_dir="ref-dir",
                uni_checkpoint="uni.pth",
                module="fake_wedetect_ref:detect",
                device="cpu",
            )
            result = client.detect(b"jpeg-bytes", "red cup", ts_req=123)
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertEqual(result.bbox, BBox(10, 20, 50, 80))
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.track_id, 42)

    def test_person_query_is_canonicalized_for_wedetect_ref(self) -> None:
        module = types.ModuleType("fake_wedetect_ref_query")
        seen = {}

        def detect(**kwargs):
            seen["query"] = kwargs["query"]
            return {
                "bbox": [10, 20, 50, 80],
                "confidence": 0.91,
                "track_id": 42,
                "query": kwargs["query"],
            }

        module.detect = detect
        sys.modules[module.__name__] = module
        try:
            client = HuggingFaceWeDetectRefClient(
                ref_model_dir="ref-dir",
                uni_checkpoint="uni.pth",
                module=f"{module.__name__}:detect",
                device="cpu",
            )
            result = client.detect(b"jpeg-bytes", "see man", ts_req=123)
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertEqual(seen["query"], "person")
        self.assertEqual(result.query, "see man")

    def test_wedetect_preflight_rejects_missing_adapter_before_download(self) -> None:
        client = HuggingFaceWeDetectRefClient(ref_model_dir="ref-dir", uni_checkpoint="uni.pth")

        with self.assertRaisesRegex(RuntimeError, "requires wedetect_ref_module"):
            client.preflight()

    def test_wedetect_preflight_calls_module_preflight(self) -> None:
        module = types.ModuleType("fake_wedetect_ref_preflight")
        called = {"preflight": False}

        def preflight():
            called["preflight"] = True

        def detect(**kwargs):
            return None

        module.preflight = preflight
        module.detect = detect
        sys.modules[module.__name__] = module
        try:
            client = HuggingFaceWeDetectRefClient(
                ref_model_dir="ref-dir",
                uni_checkpoint="uni.pth",
                module=f"{module.__name__}:detect",
                device="cpu",
            )
            artifacts = client.preflight()
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertTrue(called["preflight"])
        self.assertEqual(artifacts, ("ref-dir", "uni.pth"))

    def test_real_wedetect_ref_runtime_adapter_is_importable(self) -> None:
        module = import_module("iot_servo_tracker.server.wedetect_ref_runtime")

        self.assertTrue(callable(module.detect))

    def test_server_frame_exception_returns_empty_result(self) -> None:
        class FailingPipeline:
            def process_frame(self, *args, **kwargs):
                raise RuntimeError("decode failed")

        runtime = VisionRuntime(AppConfig(), pipeline=FailingPipeline())

        with contextlib.redirect_stderr(io.StringIO()):
            result = process_frame_safely(
                runtime,
                ts_req=123,
                query="red cup",
                frame_bytes=b"bad-jpeg",
                redetect=False,
            )

        self.assertIsNone(result.bbox)
        self.assertEqual(result.ts_req, 123)
        self.assertEqual(result.query, "red cup")

    def test_yolo_loss_falls_back_to_wedetect_redetection(self) -> None:
        class RedetectClient:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
                self.calls += 1
                return TrackingResult(
                    packet="tracking_result",
                    ts_req=ts_req,
                    ts_resp=ts_req + 1,
                    bbox=BBox(100, 100, 180, 180),
                    confidence=0.95,
                    track_id=99,
                    query=query,
                )

        class EmptyYolo:
            def track(self, *args, **kwargs):
                return []

        pipeline = build_test_pipeline(EmptyYolo(), RedetectClient())
        pipeline.locked_bbox = BBox(105, 100, 175, 180)

        first = pipeline.process_frame(ts_req=10, query="red cup", frame_bytes=b"jpeg")
        second = pipeline.process_frame(ts_req=20, query="red cup", frame_bytes=b"jpeg")

        self.assertIsNone(first.bbox)
        self.assertEqual(second.bbox, BBox(100, 100, 180, 180))
        self.assertEqual(pipeline.locked_track_id, 99)
        self.assertEqual(pipeline.yolo_lost_count, 0)
        self.assertEqual(pipeline.wedetect_client.calls, 1)

    def test_default_yolo_loss_window_waits_before_redetect(self) -> None:
        class EmptyYolo:
            def track(self, *args, **kwargs):
                return []

        redetect = EmptyRedetectClient()
        pipeline = build_test_pipeline(EmptyYolo(), redetect)
        pipeline.yolo_lost_frames = AppConfig().server.yolo_lost_frames

        for ts_req in (10, 20, 30):
            result = pipeline.process_frame(ts_req=ts_req, query="person", frame_bytes=b"jpeg")

        self.assertEqual(AppConfig().server.yolo_lost_frames, 12)
        self.assertIsNone(result.bbox)
        self.assertEqual(pipeline.yolo_lost_count, 3)
        self.assertEqual(redetect.calls, 0)

    def test_rejects_high_confidence_similar_object_switch(self) -> None:
        yolo = FakeYolo([FakeResult([([520, 220, 560, 260], 0.99, 8)])])
        redetect = EmptyRedetectClient()
        pipeline = build_test_pipeline(yolo, redetect)

        result = pipeline.process_frame(ts_req=10, query="red cup", frame_bytes=b"jpeg")

        self.assertIsNone(result.bbox)
        self.assertEqual(redetect.calls, 1)
        self.assertIn("center jumped", pipeline.last_yolo_reject_reason)

    def test_rejects_high_confidence_large_background_bbox_same_track(self) -> None:
        yolo = FakeYolo([FakeResult([([100, 80, 600, 460], 0.99, 7)])])
        redetect = EmptyRedetectClient()
        pipeline = build_test_pipeline(yolo, redetect)

        result = pipeline.process_frame(ts_req=10, query="red cup", frame_bytes=b"jpeg")

        self.assertIsNone(result.bbox)
        self.assertEqual(redetect.calls, 1)
        self.assertIn("too large", pipeline.last_yolo_reject_reason)

    def test_accepts_high_confidence_continuous_yolo_candidate(self) -> None:
        yolo = FakeYolo([FakeResult([([305, 222, 345, 262], 0.99, 7)])])
        pipeline = build_test_pipeline(yolo)

        result = pipeline.process_frame(ts_req=10, query="red cup", frame_bytes=b"jpeg")

        self.assertEqual(result.bbox, BBox(305, 222, 345, 262))
        self.assertEqual(pipeline.locked_track_id, 7)
        self.assertEqual(pipeline.wedetect_client.calls, 0)

    def test_accepts_spatially_stable_candidate_when_track_id_changes(self) -> None:
        yolo = FakeYolo([FakeResult([([336, 220, 376, 260], 0.99, 8)])])
        pipeline = build_test_pipeline(yolo)

        result = pipeline.process_frame(ts_req=10, query="person", frame_bytes=b"jpeg")

        self.assertEqual(result.bbox, BBox(336, 220, 376, 260))
        self.assertEqual(pipeline.locked_track_id, 8)
        self.assertEqual(pipeline.wedetect_client.calls, 0)

    def test_rejects_yolo_class_that_does_not_match_query(self) -> None:
        names = {0: "person", 67: "cell phone"}
        yolo = FakeYolo([FakeResult([([305, 222, 345, 262], 0.99, 7, 0)], names=names)])
        redetect = EmptyRedetectClient()
        pipeline = build_test_pipeline(yolo, redetect)

        result = pipeline.process_frame(ts_req=10, query="smartphone", frame_bytes=b"jpeg")

        self.assertIsNone(result.bbox)
        self.assertEqual(redetect.calls, 1)
        self.assertIn("does not match query", pipeline.last_yolo_reject_reason)

    def test_accepts_cell_phone_class_for_smartphone_query(self) -> None:
        names = {0: "person", 67: "cell phone"}
        yolo = FakeYolo([FakeResult([([305, 222, 345, 262], 0.99, 7, 67)], names=names)])
        pipeline = build_test_pipeline(yolo)

        result = pipeline.process_frame(ts_req=10, query="smartphone", frame_bytes=b"jpeg")

        self.assertEqual(result.bbox, BBox(305, 222, 345, 262))
        self.assertEqual(pipeline.locked_track_id, 7)
        self.assertEqual(pipeline.wedetect_client.calls, 0)

    def test_rejected_redetect_candidate_is_not_accepted_next_frame_as_new_lock(self) -> None:
        class SimilarObjectWeDetect:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
                self.calls += 1
                return TrackingResult(
                    packet="tracking_result",
                    ts_req=ts_req,
                    ts_resp=ts_req + 1,
                    bbox=BBox(520, 220, 560, 260),
                    confidence=0.99,
                    track_id=8,
                    query=query,
                )

        yolo = FakeYolo([FakeResult([([520, 220, 560, 260], 0.99, 8)])])
        wedetect = SimilarObjectWeDetect()
        pipeline = build_test_pipeline(yolo, wedetect)

        first = pipeline.process_frame(ts_req=10, query="red cup", frame_bytes=b"jpeg")
        second = pipeline.process_frame(ts_req=20, query="red cup", frame_bytes=b"jpeg")

        self.assertIsNone(first.bbox)
        self.assertIsNone(second.bbox)
        self.assertIsNone(pipeline.locked_bbox)
        self.assertEqual(pipeline.redetect_reference_bbox, BBox(300, 220, 340, 260))


if __name__ == "__main__":
    unittest.main()
