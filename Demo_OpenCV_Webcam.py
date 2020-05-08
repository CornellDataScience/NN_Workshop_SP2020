#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np

## Video Processing Modules
import os
import multiprocessing as mp
import io
from PIL import Image
from tqdm import tqdm
import time

from pipeline.capture_video import CaptureVideo
from pipeline.async_predict import AsyncPredict
from pipeline.separate_background import SeparateBackground
from pipeline.virtual_background import VirtualBackground
from pipeline.save_video import SaveVideo
from fcn import load_fcn
##


## Globals
WEIGHTS_FILE = os.path.join("weights", "fcn_weights.bin")
CPUS = 2
QUEUE_SIZE = 16
OUTPUT = "demo"
OUT_VIDEO = "demo_processed.mp4"

# Unfortunately Macs have some difficulty loading files :( so this is done manually
# We wanted the same experience for everyone.
BACKGROUND_IMAGE = "theoffice.jpg"
##

def create_pipeline():
    capture_video = CaptureVideo("demo.mp4")
    mp.set_start_method("spawn", force=True)
    predict = AsyncPredict(model_path=WEIGHTS_FILE,
                           load_model_fn=load_fcn,
                           num_cpus=CPUS,
                           queue_size=QUEUE_SIZE,
                           ordered=True
                          )
    # separate_background = SeparateBackground("vis_image")
    separate_background = VirtualBackground("vis_image", BACKGROUND_IMAGE)

    save_video = SaveVideo("vis_image", os.path.join(OUTPUT, OUT_VIDEO), capture_video.fps*0.5)

    pipeline = (capture_video |
                predict |
                separate_background |
                save_video)
    return iter(pipeline), (capture_video, predict, save_video)

def step_pipeline(pipeline, progress_bar, frame_count, others, i):
    print(i / frame_count)
    try:
        next(pipeline)
        progress_bar.UpdateBar(i + 1, frame_count)
        return
    except StopIteration:
        for item in others:
            item.cleanup()
        return 'DONE'
    except KeyboardInterrupt:
        return
    finally:
        pass
        # capture_video.cleanup()
        # predict.cleanup()
        # save_video.cleanup()


def main():

    sg.theme('Black')
    progressbar = [
        [sg.ProgressBar(1, orientation='h', size=(50, 10), key='progressbar')]
    ]
    pipeline = None
    frame_count = 1
    writer = None
    display = False
    processing = False


    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Frame('Progress',layout= progressbar)],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Process', size=(10, 1), font='Helvetica 14'),
               sg.Button('Display', size=(10,1), font='Helvetica 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    progress_bar = window['progressbar']

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = False
    start = time.time()
    not_touched = True

    while True:
        event, values = window.read(timeout=20)
        if time.time() - start >= 5 and not_touched:
            event = 'Exit'
        if event == 'Exit' or event == sg.WIN_CLOSED:
            if writer:
                writer.release()
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            return

        elif event == 'Record':
            not_touched = False
            recording = True
            if cap is None:
                cap = cv2.VideoCapture(0)
        elif event == 'Display':
            not_touched = False
            recording = False
            if processing != "DONE":
                pass
            else:
                display = True
                cap = cv2.VideoCapture(os.path.join(OUTPUT, OUT_VIDEO))
        elif event == 'Process':
            not_touched = False
            i = 0
            recording = False
            processing = True
            if writer:
                writer.release()
            cap.release()
            cap = None
            cv2.destroyAllWindows()
            if writer:
                writer.release()

            pipeline, others = create_pipeline()
            frame_count = others[0].frame_count
            while processing != "DONE":
                processing = step_pipeline(pipeline, progress_bar, frame_count, others, i)
                i += 1

            print("Done processing")

        # if not processing:
        #     i += 1
        #     processing = (step_pipeline(pipeline, progress_bar, frame_count, others, i) == "DONE")
        if display:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480), Image.BILINEAR)
                bio = io.BytesIO()
                img.save(bio, format="PNG")
                window['image'].update(data=bio.getvalue())
                # imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
                # window['image'].update(data=imgbytes)
            else:
                img = np.full((480, 640), 255)
                # this is faster, shorter and needs less includes
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                window['image'].update(data=imgbytes)
        elif recording:
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            # print("The FPS is {}".format(cap.get(cv2.CAP_PROP_FPS)))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480), Image.BILINEAR)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            if writer is None:
                writer = cv2.VideoWriter(
                        filename="demo.mp4",
                        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                        fps=0.5*cap.get(cv2.CAP_PROP_FPS),
                        frameSize=(w, h),
                        isColor=(frame.ndim == 3))
            writer.write(frame)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            # window['image'].update(data=imgbytes)
            window['image'].update(data=bio.getvalue())


main()

