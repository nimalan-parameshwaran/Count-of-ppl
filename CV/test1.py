import cv2
import numpy as np
# Choose a suitable model (e.g., YOLOv5, SSD, Faster R-CNN) and download its weights
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Example using Haar cascade
def count_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = model.detectMultiScale(gray, 1.1, 4)

    count = 0
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Optional: draw bounding boxes
        count += 1

    return count, frame
cap = cv2.VideoCapture('C:/Users/nimal/Downloads/CV/Video.mp4')

total_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count, frame = count_people(frame)
    total_count += count

    # Display the frame with count (optional)
    cv2.putText(frame, f"Total People: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
