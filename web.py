import torch
import torch.nn as nn
import torch.nn.functional as F
import tornado.ioloop
import tornado.web
import tornado.websocket
import json
import os

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device, weights_only=True))
model.eval()
print(f"CNN model loaded on {device}")

#预测
def predict(image_data):
    """image_data: 784 维灰度列表, 黑底白字 (0-255)"""
    arr = torch.tensor(image_data, dtype=torch.float32).reshape(1, 1, 28, 28) / 255.0
    arr = arr.to(device)
    with torch.no_grad():
        output = model(arr)
        pred = torch.argmax(output, dim=1).item()
    return pred



class MainWebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    async def on_message(self, message):
        try:
            data = json.loads(message)
            if "image" not in data:
                self.write_message(json.dumps({"error": "No image data provided"}))
                return
            prediction = predict(data["image"])
            self.write_message(json.dumps({"prediction": prediction}))
        except Exception as e:
            self.write_message(json.dumps({"error": str(e)}))

#静态文件服务
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

def make_app():
    return tornado.web.Application([
        (r"/ws", MainWebSocketHandler),
        (r"/", MainHandler),
    ], template_path=os.path.dirname(__file__),
       static_path=os.path.dirname(__file__))

if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    print("Server running at http://localhost:8080")
    print("WebSocket at ws://localhost:8080/ws")
    tornado.ioloop.IOLoop.current().start()
