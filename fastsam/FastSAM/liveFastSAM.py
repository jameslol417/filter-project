from fastsam import FastSAM, FastSAMPrompt
import torch
import argparse
import numpy as np
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-s.pt", help="model"
    )
    parser.add_argument(
        "--r", type=int, default="0", help="RGB R"
    )
    parser.add_argument(
        "--g", type=int, default="0", help="RGB G"
    )
    parser.add_argument(
        "--b", type=int, default="0", help="RGB B"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=256, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default="person", help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.7, help="object confidence threshold"
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
    print(device)
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
        "--withContours", type=bool, default=True, help="draw the edges of the masks"
    )
    parser.add_argument(
        "--withBoxes", type=bool, default=False, help="draw bounding boxes"
    )
    return parser.parse_args()

def nothing(x):
 pass

def main(args):

    model = FastSAM(args.model_path)

    cap = cv2.VideoCapture(2)

    # Set the interval in seconds for capturing frames

    while cap.isOpened():

        suc, frame = cap.read()

        start = time.perf_counter()

        everything_results = model(
            source=frame,
            device=args.device,
            retina_masks=True,
            imgsz=args.imgsz,
            conf=0.7,
            iou=0.9,
        )

        print(everything_results[0].masks.shape)
        print(everything_results[0].boxes.shape)
        print(everything_results[0].boxes[0].xyxy.cpu().numpy())

        if args.withBoxes:
            for box in everything_results[0].boxes:
                box = box.xyxy.cpu().numpy()[0]
                print(box)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # everything_results = list(everything_results)

        # print(everything_results[0].masks.shape)

        # end = time.perf_counter()
        # total_time = end - start
        # fps = 1 / total_time

        prompt_process = FastSAMPrompt(frame, everything_results, device=args.device)

        # # everything prompt
        # ann = prompt_process.everything_prompt()

        # # bbox prompt
        # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        # bboxes default shape [[0,0,0,0]] -> [[x1,y1,x2,y2]]
        # ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])
        # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300], [500, 500, 600, 600]])

        # # text prompt
        ann = prompt_process.text_prompt(text=args.text_prompt)

        # # point prompt
        # # points default [[0,0]] [[x1,y1],[x2,y2]]
        # # point_label default [0] [1,0] 0:background, 1:foreground
        # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

        # point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
        colors = [args.r / 255, args.g / 255, args.b / 255]
        img = prompt_process.plot_to_result1(frame,
                                             annotations=ann,
                                             withContours=args.withContours,
                                             apply_gaussian_blur=True,
                                             colors=colors)

        # cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.imshow('img', img)
        # print(cv2.getBuildInformation())
        # cv2.createTrackbar("Filters", 'img', 0, 2, nothing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)