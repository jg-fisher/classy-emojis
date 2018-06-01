from keras.models import load_model
import cv2
import numpy as np

# loading model
model = load_model('weights_v3.h5')

# starting webcam capture
cap = cv2.VideoCapture(0)

while True:
    
    # capturing webcam image
    ret, frame = cap.read()

    # preprocessing image
    frame = frame[95:600, 0:550]

    # show frame
    cv2.imshow('image', frame)

    frame = cv2.Canny(frame, 100, 200)
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    frame = frame[np.newaxis, :, :, np.newaxis]

    # model prediction
    y_prob = model.predict(frame)
    prediction = y_prob.argmax(axis=-1)
    print(prediction)

    # choosing emoji based on prediction: perfect=0, thumbsup=1, peace=2
    if prediction == 0:
        emoji = cv2.imread(r'emoji_images/perfect.png')
    elif prediction == 1:
        emoji = cv2.imread(r'emoji_images/thumbsup.png')
    elif prediction == 2:
        emoji = cv2.imread(r'emoji_images/peace.png')

    # overlaying emoji on new frame
    #emoji = cv2.resize(emoji, (0,0), fx=0.5, fy=0.5)
    #x_offset = y_offset = 50
    #x[y_offset:y_offset+emoji.shape[0], x_offset:x_offset+emoji.shape[1]] = emoji

    # show frame
    cv2.imshow('emoji', emoji)

    # end loop if exit webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close webcam
cv2.DestroyAllWindows()

