from reachy2_sdk import ReachySDK 
from reachy2_sdk.utils.utils import invert_affine_transformation_matrix
from pollen_vision.camera_wrappers.pollen_sdk_camera.pollen_sdk_camera_wrapper import PollenSDKCameraWrapper
from pollen_vision.vision_models.object_detection.owl_vit.owl_vit_wrapper import OwlVitWrapper
from pollen_vision.perception import Perception

from pollen_vision.utils import get_bboxes
from pollen_vision.utils import Annotator

from PIL import Image

#connect to the robot
reachy = ReachySDK('192.168.1.71') # replace 'localhost' with the actual IP address of your Reachy
print("Reachy is connected :", reachy.is_connected())

mujoco_mode = False
debug = True

r_cam = PollenSDKCameraWrapper(reachy)

# get the image as a np.array, and the timestamp of the image
data, _, timestamp = r_cam.get_data()

# displays the image from the RGB camera (depth camera is available by changing 'left' to 'depth')

img = data['left']
Image.fromarray(img) 

T_cam_reachy = reachy.cameras.depth.get_extrinsics()
T_reachy_cam = invert_affine_transformation_matrix(T_cam_reachy)

labels = ["cylinder", "gift", "plant", "cup"]

yolo = OwlVitWrapper() # allows to use the YOLO model for object detection
annotator = Annotator()

yolo_predictions = yolo.infer(im=img[:, :, ::-1], candidate_labels=labels, detection_threshold=0.15)
bboxes = get_bboxes(yolo_predictions)
img_annotated = annotator.annotate(im=img, detection_predictions=yolo_predictions)
Image.fromarray(img_annotated)
if debug:
    out = Image.fromarray(img_annotated)
    out.save("/exchange/detections.png")
    print("Saved to /exchange/detections.png")

perception = Perception(r_cam, T_reachy_cam, freq=40, detection_threshold=0.2)
print("The Perception object is created.")


left_obj_target, right_obj_target =  "pizza", "pasta"
obj_to_left, obj_to_right = "plant", "cup"

perception.set_tracked_objects([obj_to_left, obj_to_right, left_obj_target, right_obj_target])
perception.start(visualize=True)

reachy.turn_on()
reachy.goto_posture('default')


def get_to_waiting_pose(reachy: ReachySDK, duration: float = 2) -> None :
    r_arm_move = reachy.r_arm.goto([30,10,-15,-115,0,0,-15], duration)
    l_arm_move = reachy.l_arm.goto([30,-10,15,-115,0,0,15], duration)

    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()

    waiting_position = reachy.r_arm.forward_kinematics([30,10,-15,-115,0,0,-15])[:3,3]
    head_move = reachy.head.look_at(waiting_position[0]+0.3, 0, waiting_position[2], duration, wait = True)

    print("Reachy in waiting pose")

print("Function get_to_waiting_pose defined.")

# get_to_waiting_pose(reachy, duration = 3)



detection_threshold = 0.01
detected_objects = perception.get_objects_infos(detection_threshold)
print(detected_objects)

