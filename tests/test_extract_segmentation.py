# tests/test_extract_segmentation.py
"""Track 2: conversation segmentation (time gap + embedding jump + cache signal)."""
import numpy as np


def _topic_a():
    return np.array([1.0, 0.0] + [0.0] * 766, np.float32)


def _topic_b():
    return np.array([0.0, 1.0] + [0.0] * 766, np.float32)


class TestSegment:
    def test_two_topic_session_splits(self):
        from bespoke.extract.conversations import segment_conversation
        turns = [
            {"ts": "2026-06-02T00:00:00", "emb": _topic_a(), "cache_creation": 50, "cache_read": 0},
            {"ts": "2026-06-02T00:05:00", "emb": _topic_a(), "cache_creation": 0, "cache_read": 100},  # warm, same topic
            {"ts": "2026-06-02T02:00:00", "emb": _topic_b(), "cache_creation": 50, "cache_read": 0},   # cold miss + big gap + topic jump
        ]
        segs = segment_conversation(turns)
        assert len(segs) == 2
        assert len(segs[0]) == 2  # the two topic-A turns
        assert len(segs[1]) == 1  # topic-B starts a new conversation

    def test_warm_cache_keeps_one_segment(self):
        from bespoke.extract.conversations import segment_conversation
        turns = [
            {"ts": "2026-06-02T00:00:00", "emb": _topic_a(), "cache_creation": 50, "cache_read": 0},
            # big gap + big jump, BUT warm cache read => user was present => stay continuous
            {"ts": "2026-06-02T01:00:00", "emb": _topic_b(), "cache_creation": 0, "cache_read": 100},
        ]
        segs = segment_conversation(turns)
        assert len(segs) == 1

    def test_no_split_when_close_and_same_topic(self):
        from bespoke.extract.conversations import segment_conversation
        turns = [
            {"ts": "2026-06-02T00:00:00", "emb": _topic_a(), "cache_creation": 10, "cache_read": 0},
            {"ts": "2026-06-02T00:00:30", "emb": _topic_a(), "cache_creation": 0, "cache_read": 50},
        ]
        assert len(segment_conversation(turns)) == 1

    def test_empty(self):
        from bespoke.extract.conversations import segment_conversation
        assert segment_conversation([]) == []
