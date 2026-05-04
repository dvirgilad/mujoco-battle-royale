import numpy as np
from unittest.mock import patch
from battle_royale.infrastructure.recording.video_recorder import VideoRecorder


def test_video_recorder_can_be_constructed(tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    assert recorder is not None


def test_add_frame_stores_frames(tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    assert recorder.frame_count == 1


@patch("battle_royale.infrastructure.recording.video_recorder.mediapy")
def test_save_calls_mediapy_write_video(mock_mediapy, tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    recorder.save()
    mock_mediapy.write_video.assert_called_once()


@patch("battle_royale.infrastructure.recording.video_recorder.mediapy")
def test_save_passes_correct_fps(mock_mediapy, tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=60)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    recorder.save()
    call_kwargs = mock_mediapy.write_video.call_args
    assert call_kwargs[1].get("fps") == 60 or call_kwargs[0][2] == 60
