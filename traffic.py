import os
import random
import numpy as np
import skvideo.io
import cv2
import utils

cv2.ocl.setUseOpenCL(False) #windows 7
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    VehicleCounter)

IMAGE_DIR = "./outimg"
VIDEO_SOURCE = "cvp.mp4"
SHAPE = (720, 1280)  # HxW
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])


def train_bg_subtractor(inst, cap, num=500):
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame,learningRate=0.001)
        
        i += 1
        if i >= num:
            return cap


def main():

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
    ])

    cap = skvideo.io.vreader(VIDEO_SOURCE)

    train_bg_subtractor(bg_subtractor, cap, num=500)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            break

        _frame_number += 1
        if _frame_number % 2 != 0:
            continue

        frame_number += 1

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()


if __name__ == "__main__":

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    main()
