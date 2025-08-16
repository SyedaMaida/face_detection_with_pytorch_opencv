import cv2
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from PIL import Image

class MyCnn(nn.Module):
    def __init__(self,in_channels):
      super().__init__()
      self.features=nn.Sequential(
          nn.Conv2d(in_channels,32,kernel_size=3,padding="same"),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.MaxPool2d(2,2),
          nn.Conv2d(32,64,kernel_size=3,padding="same"),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2,2)
      )
      self.classifier=nn.Sequential(
          nn.Flatten(),
          nn.Linear(64*32*32,128),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(128,64),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(64,2)
      )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

model = MyCnn(in_channels=3)
model.load_state_dict(torch.load("mycnn.pth", map_location=torch.device("cpu")))
model.eval()

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# we are going to transform images to 128, 128
transform=transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

webcam=cv2.VideoCapture(0)

while True:
    ret,frame=webcam.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detect Face
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Face ROI
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)  # add batch dim

        # Predict
        with torch.no_grad():
            output = model(face_tensor)  # stays on CPU
            _, predicted = torch.max(output, 1)

        cv2.putText(frame, f"Class: {predicted.item()}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()