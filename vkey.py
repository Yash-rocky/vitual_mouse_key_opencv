import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import pyautogui
import math
import numpy as np

# Initialize MediaPipe, Keyboard, and Screen settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
keyboard = Controller()

# Get physical monitor resolution for cursor mapping
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False  # Prevents script from crashing if cursor hits screen corners

# Full QWERTY Keyboard Layout (Including Space, Backspace, and Enter)
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["BACKSPACE", "SPACE", "ENTER"]
]

class Button:
    def __init__(self, pos, text, size=[65, 65]):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img):
        # Draw keyboard keys
        cv2.rectangle(img, self.pos, (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), (255, 255, 255), 2)
        
        # Adjust text offset for special larger keys
        offset_x = 10 if len(self.text) == 1 else 5
        cv2.putText(img, self.text, (self.pos[0] + offset_x, self.pos[1] + 45), 
                    cv2.FONT_HERSHEY_PLAIN, 2 if len(self.text) > 1 else 3, (255, 255, 255), 2)

# Generate Layout Dynamically
button_list = []
for i in range(3): # Standard character rows
    for j, key in enumerate(keys[i]):
        button_list.append(Button([j * 75 + 30, i * 75 + 30], key))

# Add special function keys on the bottom row
button_list.append(Button([30, 255], "BACKSPACE", [180, 65]))
button_list.append(Button([225, 255], "SPACE", [330, 65]))
button_list.append(Button([570, 255], "ENTER", [150, 65]))

# Cooldown frame counter to fix rapid double-clicking/typing
debounce_counter = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Set webcam width
cap.set(4, 720)  # Set webcam height

while True:
    success, img = cap.read()
    if not success: 
        break
        
    img = cv2.flip(img, 1) # Mirror image for intuitive control
    h, w, _ = img.shape
    
    # Draw Cursor boundary box (moving hand inside this box moves cursor across the full screen)
    box_x1, box_y1, box_x2, box_y2 = 150, 150, w - 150, h - 150
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
    
    # Process frames for Hand Tracking
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Draw Virtual Keyboard Grid
    for button in button_list:
        button.draw(img)

    # Tick down the debounce cooldown timer every frame
    if debounce_counter > 0:
        debounce_counter -= 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract Index (8) and Middle (12) finger positions
            x8 = int(hand_landmarks.landmark[8].x * w)
            y8 = int(hand_landmarks.landmark[8].y * h)
            x12 = int(hand_landmarks.landmark[12].x * w)
            y12 = int(hand_landmarks.landmark[12].y * h)
            
            # --- 1. MOUSE CURSOR TRACKING ---
            # Map the smaller hand boundary box directly to your monitor screen resolution
            cursor_x = int(np.interp(x8, [box_x1, box_x2], [0, screen_w]))
            cursor_y = int(np.interp(y8, [box_y1, box_y2], [0, screen_h]))
            
            # Smoothly update physical OS cursor position
            pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
            
            # Draw a visual circle cursor directly on your index fingertip inside OpenCV window
            cv2.circle(img, (x8, y8), 10, (255, 0, 0), cv2.FILLED)
            
            # --- 2. GESTURE & ACTION LOGIC ---
            # Calculate distance between Index finger tip and Middle finger tip
            distance = math.hypot(x12 - x8, y12 - y8)
            
            # FIXED: Stricter threshold (35) ensures you must purposefully touch fingers to click
            if distance < 35:
                cv2.circle(img, (x8, y8), 15, (0, 255, 0), cv2.FILLED) # Green flash on success
                
                if debounce_counter == 0:
                    hit_virtual_key = False
                    
                    # Check if you are clicking a virtual keyboard key
                    for button in button_list:
                        bx, by = button.pos
                        bw, bh = button.size
                        
                        if bx < x8 < bx + bw and by < y8 < by + bh:
                            hit_virtual_key = True
                            
                            # Type the respective keys
                            if button.text == "SPACE":
                                keyboard.press(" ")
                            elif button.text == "BACKSPACE":
                                keyboard.press("\b")
                            elif button.text == "ENTER":
                                keyboard.press("\r")
                            else:
                                keyboard.press(button.text.lower())
                            
                            # FIXED: Higher debounce frame value (22) stops accidental multi-typing
                            debounce_counter = 22 
                            break
                    
                    # If you pinched outside the keyboard area, trigger a real OS Left Mouse Click!
                    if not hit_virtual_key:
                        pyautogui.click()
                        # FIXED: Stricter mouse cooldown stops random double clicks
                        debounce_counter = 25 
                        
    cv2.imshow("Virtual Keyboard & OS Cursor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()