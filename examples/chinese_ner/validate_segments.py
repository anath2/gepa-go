#!/usr/bin/env python3
"""Validate segmenter offsets against the source text (rune indices)."""
import json
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    text = payload["example"]["input"]["text"]
    segments = payload["output"].get("segments", [])
    for i, seg in enumerate(segments):
        start, end = seg.get("start"), seg.get("end")
        got = seg.get("text", "")
        if not isinstance(start, int) or not isinstance(end, int):
            print(json.dumps({"score": 0, "feedback": f"segment {i}: start/end must be int"}))
            return
        if start < 0 or end > len(text) or start >= end:
            print(json.dumps({"score": 0, "feedback": f"segment {i}: offsets out of range"}))
            return
        if text[start:end] != got:
            print(json.dumps({"score": 0, "feedback": f"segment {i}: text does not match text[start:end]"}))
            return
    print(json.dumps({"score": 1, "feedback": "segments valid"}))


if __name__ == "__main__":
    main()
