#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Object detection with YOLOv3
# In this exercise, you learn how to perform object detection
# with YOLOv3 on Gretchen's video stream
#

import sys
sys.path.append('..')


# Import required modules
import cv2
import argparse
import numpy as np
import hashlib

from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy

from lib.camera_v2 import Camera
from lib.ros_environment import ROSEnvironment



def loadClasses(filename):
    """ Load classes into 'classes' list and assign a random but stable color to each class. """
    global classes, COLORS

    # Load classes into an array
    try:
        with open(filename, 'r') as file:
            classes = [line.strip() for line in file.readlines()]
    except EnvironmentError:
        print("Error: cannot load classes from '{}'.".format(filename))
        quit()

    # Assign a random (but constant) color to each class
    # Method: convert first 6 hex characters of md5 hash into RGB color values
    COLORS = []
    for idx,c in zip(range(0, len(classes)), classes):
        cstr = hashlib.md5(c.encode()).hexdigest()[0:6]
        c = tuple( int(cstr[i:i+2], 16) for i in (0, 2, 4))
        COLORS.append(c)


def drawAnchorbox(frame, class_id, confidence, box):
    """ Draw an anchorbox identified by `box' onto frame and label it with the class name and confidence. """
    global classes, COLORS

    conf_str = "{:.2f}".format(confidence).lstrip('0')
    label = "{:s} ({:s})".format(classes[class_id], conf_str)
    color = COLORS[class_id]

    # Make sure we do not print outside the top/left corner of the window
    lx = max(box[0] + 5, 0)
    ly = max(box[1] + 15, 0)

    # 3D "shadow" effect: print label with black color shifted one pixel right/down, 
    #                     then print the colored label at the indented position.
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (lx+1, ly+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )

    parser.add_argument('-p', '--preview', required=False, help = 'enable preview window', type=bool, choices=[True, False])
    parser.add_argument('--isCam', required=False, help= 'enable Camera', type=bool, choices=[True, False])

    return parser.parse_args()

def main(args):
    global cfg_path, weight_path, class_name_path, classes, COLORS

    isCam = False           # TODO: fix
    hasPreview = False      # TODO: fix

    if args.isCam is True:
        isCam = True
    if args.preview is True:
        hasPreview = True

    # setup model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)

    #
    # Setup windows
    #
    cv2.namedWindow("ObjectDetection")
    if hasPreview:
        cv2.namedWindow("Preview")

    #
    # We only need to initialize the camera if we are actually going to use it
    #
    if isCam:
        # ROSEnvironment()
        # camera = Camera()
        # camera.start()

    #
    # Here we go...
    #
    while(True):
        #
        # TODO
        #
        # - if the feed comes from the camera, set 'input_image' to the camera image,
        #   otherwise load image from disk
        # - determine height, width of image to analyze
        # - perform preprocessing, inference, and result extraction the same way as
        #   in example_image_detector.py (i.e., copy-paste the code and adjust a bit)

        if isCam:
            key = cv2.waitKey(10)
            # input_image = camera.getImage()
            input_image = cv2.VideoCapture(0)
            # suc, frame = input_image.read()
            # input_image = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')
        else:
            key = cv2.waitKey()

        #
        # Load image from disk
        #
        # if args.img_path is not None:
            # input_image = Image.open(args.img_path)
        if input_image is None:
            print("Error: cannot load image '{}'.".format(args.img_path))
            quit()
        # input_image = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # print("Image Type: ", type(input_image))
        # print("Image Type: ", input_image.dtype)

        input_image = input_image.convert("RGB")
        # print("Image Type: ", input_image.format)

        # width = input_image.shape[1]
        # height = input_image.shape[0]
        #if frame is None:
        #    frame = input_image

        everything_results = model(
            input,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou
        )
        bboxes = None
        points = None
        point_label = None
        prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
        if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
        elif args.text_prompt != None:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
            points = args.point_prompt
            point_label = args.point_label
        else:
            ann = prompt_process.everything_prompt()
        prompt_process.plot(
            annotations=ann,
            output_path=args.output + args.img_path.split("/")[-1],
            bboxes=bboxes,
            points=points,
            point_label=point_label,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )

        #
        # Preview: show all anchorboxes with a total confidence > conf_threshold
        #
        # preview = input_image.copy()
        # for idx, classid in enumerate(class_ids):
        #     drawAnchorbox(preview, classid, confidence_values[idx], bounding_boxes[idx])



        #
        # Display results
        #
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Preview", preview[..., ::-1])  # swap RGB -> BGR
        cv2.imshow("ObjectDetection", input_image[..., ::-1])  # idem
        # key = cv2.waitKey()
        # cv2.imshow('frame', frame)


        if key > 0:
            break

    cv2.destroyAllWindows()


#
# Program entry point when started directly
#
if __name__ == '__main__':
    args = parse_args()
    main(args)
