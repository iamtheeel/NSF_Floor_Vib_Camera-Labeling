import cv2
import pytesseract

class timeWith_ms():
    def __init__(self, frameTime_ms):
        self.prev_time = "hh:mm:ss"
        self.frames_since_rollover = 0
        self.first_rollover_detected = False
        self.frameTime_ms = frameTime_ms
        self.ms_since_last_rollover = 0
        self.hasRun = False

    def isRollover(self, curTime_str):
        if self.hasRun == False:
            self.hasRun = True
            self.prev_time = curTime_str
        if curTime_str != self.prev_time:
            if not self.first_rollover_detected:
                self.first_rollover_detected = True
                print(f"First OCR rollover at OCR time: {curTime_str}")

            self.frames_since_rollover = 0
            self.prev_time = curTime_str
            return True
        if self.frames_since_rollover >= 30:
                self.ms_since_last_rollover += 10
        if self.frames_since_rollover >= 32:
            print("**!!!Frame time severely off!!!**")
            exit()
        return False
    #call calc_ms as the external call
    def calc_ms(self, curTime_str, display = False):
        self.isRollover(curTime_str)
        
        self.frames_since_rollover += 1

        if not self.first_rollover_detected:
            if display == True:
                print(f"Real time : {curTime_str}")
            #TODO: backcalculate
            return (f"{curTime_str}.000")

        if self.frames_since_rollover <= 30:
            self.ms_since_last_rollover = ((self.frames_since_rollover - 1) * self.frameTime_ms) % 1000

        display_time = f"{self.prev_time}.{int(self.ms_since_last_rollover):03d}"
        if display == True:
            print(f"Real time: {display_time}")
        return display_time


