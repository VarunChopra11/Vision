import cv2
import numpy as np
import os
import time

KNOWN_WIDTHS = {
    "person": 0.5,  # meters
    "car": 2,  # meters
    # Add more objects with their known widths if needed
}

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath, focalLength=500):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.focalLength = focalLength

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.objects_within_distance = {}  # Dictionary to store objects within distance

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        # compute and return the distance from the object to the camera
        return (knownWidth * focalLength) / perWidth

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error Opening Video File...")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            classLabelIDs, confidences, bboxs = self.net.detect(frame, confThreshold=0.4)

            if len(classLabelIDs) != 0:
                current_time = time.time()
                for classLabelID, confidence, bbox in zip(classLabelIDs, confidences, bboxs):
                    bbox = list(map(int, bbox))
                    classLabel = self.classesList[classLabelID[0]]

                    # Calculate distance
                    if classLabel in KNOWN_WIDTHS:
                        width = bbox[2] - bbox[0]
                        distance = self.distance_to_camera(KNOWN_WIDTHS[classLabel], self.focalLength, width)
                        distance_text = "Distance: {:.2f}m".format(distance)
                        if distance < 0.3:
                            if classLabelID[0] not in self.objects_within_distance:
                                print(f"{classLabel} detected within 0.3m.")
                                self.objects_within_distance[classLabelID[0]] = current_time
                            else:
                                if current_time - self.objects_within_distance[classLabelID[0]] >= 1:
                                    self.objects_within_distance[classLabelID[0]] = current_time
                        else:
                            if classLabelID[0] in self.objects_within_distance:
                                if distance > 0.5:
                                    del self.objects_within_distance[classLabelID[0]]

                    else:
                        distance_text = "Distance: Unknown"

                    # Draw rectangle and text
                    color = [int(c) for c in self.colorList[classLabelID[0]]]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=1)
                    cv2.putText(frame, distance_text, (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                    cv2.putText(frame, "{}: {:.2f}".format(classLabel, confidence),
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    videoPath = 0  # For webcam, replace video path address with '0'.
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    focalLength = 500  # Adjust this value according to your camera setup

    detector = Detector(videoPath, configPath, modelPath, classesPath, focalLength)
    detector.onVideo()


if __name__ == '__main__':
    main()
