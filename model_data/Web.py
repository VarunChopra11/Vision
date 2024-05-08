import os
import cv2 
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

st.title('Obstacle Detection System')

MODEL_DATA_PATH = "model_data"
configPath = os.path.join(MODEL_DATA_PATH, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join(MODEL_DATA_PATH, "frozen_inference_graph.pb")
classesPath = os.path.join(MODEL_DATA_PATH, "coco.names")

net = cv2.dnn_DetectionModel(modelPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def read_classes():
    with open(classesPath, 'r') as f:
        classesList = f.read().splitlines()

    classesList.insert(0, '__Background__')
    return classesList

def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

def detect(frame, confThreshold=0.4):
    classIDs, confidences, bboxs = net.detect(frame, confThreshold=confThreshold)
    return classIDs, confidences, bboxs 

classesList = read_classes()
COLORS = np.random.uniform(low=0, high=255, size=(len(classesList), 3))
known_width = 10
focal_length = 280

class VideoTransformer(VideoProcessorBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        classIDs, confidences, bboxs = detect(img)

        if len(classIDs) > 0:
            for classID, confidence, bbox in zip(classIDs, confidences, bboxs):
                classLabel = classesList[int(classID)]
                color = COLORS[int(classID)]
                x, y, w, h = bbox

                display_text = f"{classLabel}: {confidence:.2f}"
                cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=1)
                cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                object_width = w
                distance = calculate_distance(known_width, focal_length, object_width)
                cv2.putText(img, f"Distance: {distance:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        return img

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
)
