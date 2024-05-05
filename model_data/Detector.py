import cv2
import numpy as np
import time

np.random.seed(20)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.known_width = 10
        self.focal_length = 280
        self.persons_detected = 0
        self.chairs_detected = 0
        self.tables_detected = 0
        self.top_objects = []

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

    def calculate_distance(self, known_width, focal_length, per_width):
        return (known_width * focal_length) / per_width

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error Opening Video File...")
            return

        (success, image) = cap.read()

        self.persons_detected = 0
        self.chairs_detected = 0
        self.tables_detected = 0

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)
            
            self.persons_detected = 0
            self.chairs_detected = 0
            self.tables_detected = 0

            object_confidences = {}

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]

                    if classLabel == 'person':
                        self.persons_detected += 1
                    if classLabel == 'chair':
                        self.chairs_detected += 1
                    if classLabel == 'table':
                        self.tables_detected += 1

                    if classLabel not in object_confidences:
                        object_confidences[classLabel] = classConfidence
                    else:
                        object_confidences[classLabel] = max(object_confidences[classLabel], classConfidence)

                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    object_width = w
                    distance = self.calculate_distance(known_width=10, focal_length=self.focal_length, per_width=object_width)
                    cv2.putText(image, f"Distance: {distance:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            sorted_objects = sorted(object_confidences.items(), key=lambda x: x[1], reverse=True)
            self.top_objects = [obj[0] for obj in sorted_objects[:5]]


            # print(self.persons_detected)  # Number of persons detected.
            # print(self.chairs_detected)   # Number of chairs detected.    
            # print(self.tables_detected)   # Number of tables detected.
            # print(self.top_objects)       # 5 objects detected with max confidence.


            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows()
