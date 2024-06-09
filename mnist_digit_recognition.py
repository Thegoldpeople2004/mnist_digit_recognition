import tkinter as tk

import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model

# 加载预训练模型
model = load_model('models/mnist_cnn_model.keras')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_clear = tk.Button(self, text="清除", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(self, text="预测", command=self.predict_digit)
        self.button_predict.pack()

        self.label = tk.Label(self, text="请在白板上画一个数字", font=("Helvetica", 18))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = PIL.Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="请在白板上画一个数字")

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.line([x1, y1, x2, y2], fill='black', width=10)

    def predict_digit(self):
        self.image = self.image.resize((28, 28))
        self.image = ImageOps.invert(self.image)
        img_array = np.array(self.image)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        self.label.config(text=f'预测结果: {predicted_class[0]}')

if __name__ == "__main__":
    app = App()
    app.mainloop()
