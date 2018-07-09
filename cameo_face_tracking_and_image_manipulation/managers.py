import cv2
import numpy
import time


class CaptureManager(object):
    def __init__(self, capture, preview_window_manager = None, should_mirror_preview = False):
        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview
        self._capture = capture
        self._channel = 0
        self._entered_frame = False
        self._frame = None
        self._image_file_name = None
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None
        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def is_writing_image(self):
        return self._image_file_name is not None

    @property
    def is_writing_video(self):
        return self._video_file_name is not None

    def enter_frame(self):
        """Capture the next frame, if any"""

        # But first, check that any previous frame was exited
        assert not self._entered_frame, \
            "previous enter_frame() had no matching exit_frame()"

        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        """Draw to the window. Write to files. Release the frame"""

        # Check whether any grabbed frame is retrievable
        # The getter may retrieve and cache the frame
        if self.frame is None:
            self._entered_frame = False
            return

        # Update the FPS estimate and related variables
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # Draw to the window, if any.
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        # Write to the image file, if any
        if self.is_writing_image:
            cv2.imwrite(self._image_file_name, self._frame)
            self._image_file_name = None

        # Write to the video file, if any
        self._write_video_frame()

        # Release the frame
        self._frame = None
        self._entered_frame = False

    def write_image(self, filename):
        """Write the next exited frame to an image file."""

        self._image_file_name = filename

    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """Start writing exited frames to a video file."""

        self._video_file_name = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        """Stop writing exited frames to a video file."""
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frames_elapsed < 20:
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(self._video_file_name, self._video_encoding, fps, size)

        self._video_writer.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypress_callback = keypress_callback

        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._window_name)

        self._is_window_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)

        if self.keypress_callback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypress_callback(keycode)