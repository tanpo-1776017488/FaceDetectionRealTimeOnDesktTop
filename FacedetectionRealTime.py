import cv2
from facenet_pytorch import MTCNN
from PIL import Image,ImageDraw
import numpy as np

mtcnn=MTCNN(keep_all=True,device="cuda")
cap=cv2.VideoCapture(0)
i =1
while True:
    ret,frame=cap.read()
    if not ret:
        break
    #convert BGR format to RGB format
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #Convert np.array format to PIL Image format
    frame=Image.fromarray(frame)
    boxes,probs,points=mtcnn.detect(frame,landmarks=True)

    
    draw=ImageDraw.Draw(frame)
    
    #if system didn't catch the face
    try:
        for i,(box,point) in enumerate(zip(boxes,points)):
            draw.rectangle(box.tolist(),width=2)
    except:
        print("error occur")
    frame=np.array(frame)
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    #mirror mode
    frame=cv2.flip(frame,1)
    cv2.imshow('video',frame)

    if cv2.waitKey(33)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# cap=cv2.VideoCapture(0)

# while True:
#     ret,frame=cap.read()
#     frame=cv2.flip(frame,1)
#     cv2.imshow('video',frame)

#     if cv2.waitKey(33)&0xFF ==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
