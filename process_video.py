import os
from tqdm import tqdm
import multiprocessing as mp

from pipeline.capture_video import CaptureVideo
from pipeline.async_predict import AsyncPredict
from pipeline.separate_background import SeparateBackground
from pipeline.save_video import SaveVideo
from fcn import load_fcn


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="FCN Zoom video processing pipeline")
    ap.add_argument("-i", "--input", default="0",
                    help="path to input video file, image frames directory or camera identifier (default: camera 0)")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ov", "--out-video", default=True,
                    help="output video file name")
    ap.add_argument("--fps", type=int, default=None,
                    help="overwrite fps for output video or if it is unknown for image frames directory")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="display progress")
    ap.add_argument("-d", "--display", action="store_true",
                    help="display video")
    ap.add_argument("-sb", "--separate-background", action="store_true",
                    help="separate background")

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
    import sys
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # First create video capture
    if args.input.isdigit():
        capture_video = CaptureVideo(int(args.input))
    elif os.path.isfile(args.input):
        capture_video = CaptureVideo(args.input)
    else:
        print("Invalid Type put in!")
        sys.exit(1)

    if args.single_process:
        predict = None
    else:
        mp.set_start_method("spawn", force=True)
        predict = AsyncPredict(model_path=args.weights_file,
                               load_model_fn=load_fcn,
                               num_cpus=args.cpus,
                               queue_size=args.queue_size,
                               ordered=True
                              )

    separate_background = SeparateBackground("vis_image")

    save_video = SaveVideo("vis_image", os.path.join(args.output, args.out_video),
                           capture_video.fps if args.fps is None else args.fps)

    pipeline = (capture_video |
                predict |
                separate_background |
                save_video)

    try:
        for _ in tqdm(pipeline, total=capture_video.frame_count if capture_video.frame_count > 0 else None, disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        # Clean up of file handles etc.
        if isinstance(predict, CaptureVideo):
            capture_video.cleanup()
        if isinstance(predict, AsyncPredict):
            predict.cleanup()
        if save_video:
            save_video.cleanup()
    




if __name__ == "__main__":
    args = parse_args()
    main(args)

