from neuralNet import NeuralNet
import cv2 as cv
import numpy as np

model = NeuralNet()

def handleImage(image):
    img_bytes = np.fromfile(image, np.uint8)
    imageBGR = cv.imdecode(img_bytes, cv.IMREAD_COLOR)
    imageRGB = cv.cvtColor(imageBGR , cv.COLOR_BGR2RGB)
    res = cv.resize(imageRGB, dsize=(28, 28), interpolation=cv.INTER_CUBIC)
    pixels = np.array(res).reshape(-1, 28, 28, 3)
    analysisResult = model.predict(pixels)
    return analysisResult

