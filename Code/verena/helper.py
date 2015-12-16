import mahotas
import numpy as np
import matplotlib.pyplot as plt

def createOverlayImage(image, label):
    red = np.copy(image)
    red[np.nonzero(label)] = 255
    greenAndBlue = np.copy(image)
    greenAndBlue[np.nonzero(label)] = 0
    return mahotas.as_rgb(red, greenAndBlue, greenAndBlue)
