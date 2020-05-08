import os
from tqdm import tqdm
import multiprocessing as mp

from pipeline.capture_video import CaptureVideo
from pipeline.async_predict import AsyncPredict
from pipeline.separate_background import SeparateBackground
from pipeline.save_video import SaveVideo


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Detectron2 video processing pipeline")
    ap.add_argument("-i", "--input", default="0",
                    help="path to input video file, image frames directory or camera identifier (default: camera 0)")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ov", "--out-video", default=None,
                    help="output video file name")
    ap.add_argument("--fps", type=int, default=None,
                    help="overwrite fps for output video or if it is unknown for image frames directory")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="display progress")
    ap.add_argument("-d", "--display", action="store_true",
                    help="display video")
    ap.add_argument("-sb", "--separate-background", action="store_true",
                    help="separate background")
    ap.add_argument("-tp", "--track-pose", action="store_true",
                    help="track pose")

    # Detectron settings
    ap.add_argument("--weights-file", default=None,
                    help="path to model weights file")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="minimum score for instance predictions to be shown (default: 0.5)")


    # Mutliprocessing settings
    ap.add_argument("--cpus", type=int, default=0,
                    help="number of CPUs (default: 0)")
    ap.add_argument("--queue-size", type=int, default=3,
                    help="queue size per process (default: 3)")
    ap.add_argument("--single-process", action="store_true",
                    help="force the pipeline to run in a single process")

    return ap.parse_args()


def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)

