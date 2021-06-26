from fastapi import APIRouter, File, UploadFile, responses
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

router = APIRouter(
    tags=['Image Object Detection']
)

CONFIDENCE_THRESHOLD=0.2
NMS_THRESHOLD=0.4
net = cv2.dnn.readNet("ai_model/yolov4/yolov4.weights", "ai_model/yolov4/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def yolov4_detect(image):
    preds = []
    img = np.asarray(image)[..., :3]
    img = img[:, :, ::-1]

    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # print(classes, scores, boxes)
    for cls, score, box in zip(classes, scores, boxes):
        preds.append({'label': int(cls[0]), 'confidence': int(score[0]), "bouding_box": box.tolist()})
        print(preds)
    return preds

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@router.post("/object_detect")
async def object_detect(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = yolov4_detect(image)
    return responses.JSONResponse(content={'result': prediction})