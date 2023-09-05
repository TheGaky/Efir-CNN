from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file = r"C:\Users\yakovgay\Downloads\gaky_shit\photo_2023-09-05_14-18-20.jpg"

image = Image.open(file)
image = Image.Image.getdata(image)

image = np.array(image)


plt.imshow(image.reshape(720, 1280, 3))
plt.show()
