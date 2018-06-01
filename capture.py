import cv2
import keyboard

cap = cv2.VideoCapture(0)

# 0 = perfect
# 1 = peace
# 2 = thumbsup 
# 3 = thumbsdown

frame_num = 0

with open('classes.csv', 'a') as f:
    while frame_num < 1501:
    
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        cv2.imshow('image', frame)
    
        if keyboard.is_pressed('space bar'):
            cv2.imwrite(r'./images/thumbsdown_{}.jpg'.format(frame_num), gray)
            f.write('3\n')
            print('written {}'.format(frame_num))
            frame_num += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            f.close()
            break

cap.release()
cv2.destroyAllWindows()
