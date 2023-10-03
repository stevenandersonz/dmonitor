import cv2
import numpy as np
import onnx
import onnxruntime

#LOAD MODEL FROM COMMA.AI
# https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md
onnx_model = onnx.load("dmonitoring_model.onnx")
ort_session = onnxruntime.InferenceSession("dmonitoring_model.onnx",providers=['CPUExecutionProvider'])

#UTILS to parse the output of the networ

REG_SCALE = 0.25
def face_orientation_from_net(angles_desc, pos_desc, rpy_calib=[0.0,0.0,0.0], W=720, H=990):
    # the output of these angles are in device frame
    # so from driver's perspective, pitch is up and yaw is right
    pitch_net, yaw_net, roll_net = angles_desc

    face_pixel_position = ((pos_desc[0]+0.5)*W, (pos_desc[1]+0.5)*H)
    yaw_focal_angle = atan2(face_pixel_position[0] - W//2, EFL)
    pitch_focal_angle = atan2(face_pixel_position[1] - H//2, EFL)

    pitch = pitch_net + pitch_focal_angle
    yaw = -yaw_net + yaw_focal_angle

    # no calib for roll
    pitch -= rpy_calib[1]
    yaw -= rpy_calib[2]
    return roll_net, pitch, yaw

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DriverStateResult:
    def __init__(self):
        self.face_orientation = [0.0, 0.0, 0.0]
        self.face_orientation_std = [0.0, 0.0, 0.0]
        self.face_position = [0.0, 0.0]
        self.face_position_std = [0.0, 0.0]
        self.ready_prob = [0.0, 0.0, 0.0, 0.0]
        self.not_ready_prob = [0.0, 0.0]
        self.face_prob = 0.0
        self.face_size = 0.0
        self.left_eye_prob = 0.0
        self.right_eye_prob = 0.0
        self.left_blink_prob = 0.0
        self.right_blink_prob = 0.0
        self.sunglasses_prob = 0.0
        self.occluded_prob = 0.0

def parse_driver_data(d_state, s, out_idx_offset):
    for i in range(3):
        d_state.face_orientation[i] = s[out_idx_offset + i] * REG_SCALE
        d_state.face_orientation_std[i] = np.exp(s[out_idx_offset + 6 + i])
        
    for i in range(2):
        d_state.face_position[i] = s[out_idx_offset + 3 + i] * REG_SCALE
        d_state.face_position_std[i] = np.exp(s[out_idx_offset + 9 + i])

    #face size?
    d_state.face_size = s[out_idx_offset+11] 

    for i in range(4):
        d_state.ready_prob[i] = sigmoid(s[out_idx_offset + 35 + i])
        
    for i in range(2):
        d_state.not_ready_prob[i] = sigmoid(s[out_idx_offset + 39 + i])

    d_state.face_prob = sigmoid(s[out_idx_offset + 12])
    d_state.left_eye_prob = sigmoid(s[out_idx_offset + 21])
    d_state.right_eye_prob = sigmoid(s[out_idx_offset + 30])
    d_state.left_blink_prob = sigmoid(s[out_idx_offset + 31])
    d_state.right_blink_prob = sigmoid(s[out_idx_offset + 32])
    d_state.sunglasses_prob = sigmoid(s[out_idx_offset + 33])
    d_state.occluded_prob = sigmoid(s[out_idx_offset + 34])

#MODEL can detect two passengers.
driver_state_r = DriverStateResult()
driver_state_l = DriverStateResult()




# DETECT DRIVER 
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize and convert frame to YUV
    frame_resized = cv2.resize(frame, (1440, 960))
    frame_yuv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
    y_channel = frame_yuv[:, :, 0]

    # Normalize Y channel to float32
    y_normalized = y_channel.astype(np.float32) / 255.0

    # Run ONNX model
    ort_inputs = {
        ort_session.get_inputs()[0].name: y_normalized.reshape(1, 1440*960),
        ort_session.get_inputs()[1].name: np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3)}
    ort_outs = ort_session.run(None, ort_inputs)
    #NOTE: Model output for two person in the car
    parse_driver_data(driver_state_r, ort_outs[0].squeeze(), 0)
    parse_driver_data(driver_state_l, ort_outs[0].squeeze(), 41)
    print(driver_state_r.face_size)
    label = f"Face {'Detected' if driver_state_r.face_prob >= 0.5 else 'Missing'} - {driver_state_r.face_prob:.2f}"
    # TODO: doesn't work i'm gessing due to image calibration or how i'm sitting
    label2 = f"Face Position y: {driver_state_r.face_position[0]} x:{driver_state_r.face_position[1]}"
    label3 = f"R Eye: {driver_state_r.right_eye_prob:.2f} L Eye:{driver_state_r.left_eye_prob:.2f}"

    # detecting the position of the face in the frame
    H, W, _ = frame.shape
    face_x, face_y = ((driver_state_r.face_position[0]+0.5)*W, (driver_state_r.face_position[1]+0.5)*H)
    print(face_x)
    print(face_y)
    cv2.circle(frame, (int(face_x), int(face_y)), 5, (0, 255, 0), -1)
    # # I think this goes as:
    cv2.putText(frame, f'{label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{label2}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{label3}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'sunglases: {driver_state_r.sunglasses_prob:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()