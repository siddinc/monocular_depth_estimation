import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json


def normalize_image(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img


def preprocess_image(img):
    image = imutils.resize(img, height=128)
    image = image[:, 21:149].astype("float")
    image = normalize_image(image)
    image = np.reshape(image, (1, 128, 128, 3))
    return image

def display_image(img):
    resized_img = imutils.resize(img, height=256)
    resized_img = img_as_ubyte(resized_img)
    img_colorized = cv2.applyColorMap(resized_img, cv2.COLORMAP_HSV)
    cv2.imshow("depth_map", img_colorized)



def load_model():
    with open("../models/model.json", "r") as f:
        m = f.read()
        model = model_from_json(m)
        model.load_weights("../weights.h5")
    return model


def main():
    loaded_model = load_model()
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            frame = cv2.flip(frame, 1)
            cv2.imshow("cam_feed", frame)
            preprocessed_frame = preprocess_image(frame)
            pred = loaded_model.predict(preprocessed_frame)
            pred = np.squeeze(pred)
            display_image(pred)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
