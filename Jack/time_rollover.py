
import cv2
import pytesseract

def getTime(frame):
    dateTime_img = frame[0:45, 0:384]  # Crop the date and time area
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    dateTime_img_bw = cv2.bitwise_not(dateTime_img_bw)  # Invert the colors
    dateTime_outPut = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    timeStr_num = 5
    return dateTime_outPut['text'][timeStr_num]

current_time = getTime(frame)

    if prev_time is None:
        prev_time = current_time

    if current_time != prev_time:
        print(f"Frames since last rollover: {frames_since_last}")
        frames_since_last = 0  # Start counting for the new time value
        prev_time = current_time