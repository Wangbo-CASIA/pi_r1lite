# 该脚本的功能是接收外部的状态机，并进行内部状态的切换，并通过全局变量也可以自己在模型推理结束以后也进行内部状态切换


import cv2
import time
import argparse
import numpy as np
from dataclasses import dataclass
from multiprocessing import Value, RawArray

from teleop_rec4.post_process import interp_SE3_sep, interp_linear
from scipy.spatial.transform import Rotation
import rospy
from fast_msgs.msg import wbc_state
from vla_msgs.msg import record_eval_msg
import ctypes
from functools import partial
import pygame
from fast_msgs.srv import stateMachine, stateMachineResponse
from enum import Enum
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import threading

import signal
import sys

def shutdown_handler(sig, frame):
    rospy.signal_shutdown("Received SIGINT")
    sys.exit(0)



class NodeStatus(Enum):
    IDLE = "idle"             # 空闲状态
    IN_PROGRESS = "in_progress"  # 任务正在进行
    IN_PROGRESS_STAGE0 = "in_progress_stage0"  # 任务正在进行
    IN_PROGRESS_STAGE1 = "in_progress_stage1"  # 任务正在进行
    DONE = "done"             # 任务完成
    ERROR = "error"           # 错误状态
    
vla_status = NodeStatus.IDLE
status_lock = threading.Lock()
status_pub = None




def publish_current_status():
    # 该函数的作用是将当前状态通过 ROS 话题发布出去，以便其他节点能够订阅并了解当前状态。
    # 它首先检查 status_pub 是否已初始化，如果没有则直接返回。
    # 然后创建一个 String 消息，将当前状态的字符串值赋给消息的数据字段，
    # 并通过 status_pub 发布该消息。
    # 最后，使用 rospy.loginfo 输出日志，记录发布的状态信息。
    # 这种机制允许系统中的其他组件根据当前状态做出相应的反应，实现不同模块之间的协同工作。
    global status_pub
    global vla_status 
    if status_pub is None:
        return
    msg = String()
    msg.data = vla_status.value
    status_pub.publish(msg)
    rospy.loginfo(f"Broadcast status: {msg.data}")


def vla_state_sub_callback(msg):
    """订阅 '/vla_state_sub'，根据收到的命令改变状态
    外部状态的逻辑是：
    # 最开始外部啥msg都没有 或者者发一个reset 让它变成idle
    # 发一个control 让它变成in_progress 这个时候record_fn会开始采集数据并发布消息；
    # 模型1开始执行，并时刻进行判断是否结束变成done；外部接收到done以后就会改成idle或者下一个状态
    # 重复这个过程...
    """

    global vla_status
    command = msg.data
    rospy.loginfo(f"Received state command via topic: {command}")
    with status_lock:
        if command == 'query':
            pass
        elif command == 'control':
            vla_status = NodeStatus.IN_PROGRESS
        elif command == 'controlstage0':
            vla_status = NodeStatus.IN_PROGRESS_STAGE0
        elif command == 'controlstage1':
            vla_status = NodeStatus.IN_PROGRESS_STAGE1
        elif command == 'stop':
            vla_status = NodeStatus.DONE
        elif command == 'reset':
            vla_status = NodeStatus.IDLE
        else:
            rospy.logwarn(f"Unknown command: {command}")
        publish_current_status()


@dataclass
class PlanConfig(object):
    # 相机配置
    e2h_camera_index: int = 1  # set to -1 to disable
    eih_camera_index: int = 1  # set to -1 to disable

    record_dt: float = 0.1


def put_text(image: np.ndarray, text=None, small_text=None):
    text = text or []
    small_text = small_text or []
    H = image.shape[0]
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    return image


def action_interp(
    train_poses,
    train_openness, 
    train_timesteps,
    query_time
):
    if query_time >= train_timesteps[-1]:
        query_pose = train_poses[-1]
        query_openness = train_openness[-1]
    elif query_time <= train_timesteps[0]:
        query_pose = train_poses[0]
        query_openness = train_openness[0]
    else:
        bin_index = np.digitize(query_time, train_timesteps)
        l_index = bin_index - 1  # left index
        query_pose = interp_SE3_sep(
            T0=train_poses[l_index],
            T1=train_poses[l_index+1],
            t0=train_timesteps[l_index],
            t1=train_timesteps[l_index+1],
            t=query_time
        )
        query_openness = interp_linear(
            q0=train_openness[l_index],
            q1=train_openness[l_index+1],
            t0=train_timesteps[l_index],
            t1=train_timesteps[l_index+1],
            t=query_time
        )
    return query_pose, query_openness

def quaternion_to_matrix(q):
    """使用SciPy将四元数转换为旋转矩阵"""
    # SciPy的四元数格式是[x, y, z, w]
    r = Rotation.from_quat([q[0], q[1], q[2], q[3]])

    return r.as_matrix()


        
def handle_state_control_service(req):
    global vla_status
    """
    服务回调：忽略所有传入的命令参数（control/stop/reset等均无效），
    仅返回当前状态字符串。
    """
    # 无论 req.command 是什么，都只返回当前状态
    return stateMachineResponse(message=vla_status.value)


def set_status(status):
    global vla_status
    with status_lock:
        vla_status = status
        publish_current_status()



def record_fn(ctrl_state, plan_config: PlanConfig, is_head_recived, is_righthand_recived, is_eepose_recived, head_img_shared, head_img_time_shared, right_img_shared,ee_pose_shared, ee_pose_time_shared, gripper_shared,state_publisher):

    global vla_status
    rate = rospy.Rate(10)

    def wait_for_value(val, name):
        while not rospy.is_shutdown():
            if val.value:
                return
            print(f"[INFO] Waiting for {name} from ros...")
            rate.sleep()

    # 等待所有必要数据到位  只有当这三个共享变量都变为 1（即数据都已收到），主循环才会正式进入数据处理和发布阶段，保证后续处理不会因为数据未准备好而出错。
    wait_for_value(is_righthand_recived, "right_img")
    wait_for_value(is_eepose_recived, "ee_pose")
    wait_for_value(is_head_recived, "head_img")

    history_times = []
    history_poses = []
    history_openness = []

    en_e2h = plan_config.e2h_camera_index >= 0
    en_eih = plan_config.eih_camera_index >= 0

    pygame.init()
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("Debug View") # 初始化 Pygame 图形界面，并创建一个用于显示调试信息的窗口


    is_first = True
    start_time = 2746710249.0890572
    end_time = start_time

    def reset_history():
        # 在任务完成或需要重置时，清空历史数据和状态，恢复到初始状态。
        nonlocal history_times, history_poses, history_openness, is_first, start_time, end_time
        is_first = True # 标记为“首次进入”状态，后续流程会重新计时
        start_time = 2746710249.0890572
        end_time = start_time # start_time 和 end_time 设为初始时间（2746710249.0890572）：用于后续计时逻辑的重置
        history_times = []
        history_poses = []
        history_openness = []

    def update_start_time_for_active_status():
        nonlocal is_first, start_time
        if is_first: # 只有首次进入该阶段时才执行下面的操作
            start_time = time.time() # 标记已进入过该阶段，避免重复计时。
            is_first = False # 标记已进入过该阶段

    try:
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rospy.signal_shutdown("Debug window closed")
                    break

            # 这段主要是用于调试显示的
            debug_bgr = []
            win_names = []
            e2h_frame = {}
            eih_frame = {}

            if en_e2h:
                e2h_frame["bgr"] = head_img_shared.copy()
                e2h_frame["timestep"] = head_img_time_shared.value
                debug_bgr.append(e2h_frame["bgr"])
                win_names.append("e2h")
            if en_eih:
                eih_frame["bgr"] = right_img_shared.copy()
                eih_frame["timestep"] = head_img_time_shared.value
                debug_bgr.append(eih_frame["bgr"])
                win_names.append("eih")

            if not debug_bgr:
                rospy.logwarn_throttle(2.0, "No camera is enabled.")
                rate.sleep()
                continue

            debug_bgr = np.concatenate(debug_bgr, axis=1)
            win_names = " | ".join(win_names)
            put_text(debug_bgr, ["Paused" if ctrl_state.value else "Playing"])
            debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(debug_rgb.transpose(1, 0, 2))
            screen.blit(surf, (0, 0))
            pygame.display.flip()


            if vla_status in (
                NodeStatus.IN_PROGRESS,
                NodeStatus.IN_PROGRESS_STAGE0,
                NodeStatus.IN_PROGRESS_STAGE1,
            ):
                update_start_time_for_active_status()

            # 根据当前状态决定是否继续采集数据和发布消息；如果处于 IDLE 状态，则跳过数据处理和发布，直接进入下一轮循环等待状态改变。
            if vla_status == NodeStatus.IDLE:
                rospy.loginfo_throttle(2.0, "IDLE****************************")
                rate.sleep()
                continue
            if vla_status == NodeStatus.DONE:
                reset_history() # Done是我自己根据任务是否结束发的
                rospy.loginfo_throttle(2.0, "Done****************************")
                rate.sleep()
                continue

            # 采集数据
            ee_pose = np.asarray(ee_pose_shared.copy())
            norm_openness = float(gripper_shared.value)
            timestep = float(ee_pose_time_shared.value)
            current_state = {
                "base_ee_T": ee_pose,
                "timestep": timestep,
                "norm_openness": norm_openness
            }
            end_time = time.time()

            history_times.append(current_state["timestep"])
            history_poses.append(current_state["base_ee_T"])
            history_openness.append(current_state["norm_openness"])
            while len(history_poses) > 10:
                history_poses.pop(0)
                history_times.pop(0)
                history_openness.pop(0)



            # 只在IN_PROGRESS/IN_PROGRESS_STAGE1时发布消息
            if vla_status in (NodeStatus.IN_PROGRESS, NodeStatus.IN_PROGRESS_STAGE1):
                train_times = np.asarray(history_times) # 末端pose的时间戳
                train_poses = np.asarray(history_poses)
                train_openness = np.asarray(history_openness)
                # 让发布出去的图像和机器人状态在时间上尽量对齐，减少相机帧和机械臂状态之间的时间偏差
                query_time = head_img_time_shared.value # 使用头部相机时间戳作为查询时间
                query_pose, query_openness = action_interp(
                    train_poses=train_poses,
                    train_openness=train_openness,
                    train_timesteps=train_times,
                    query_time=query_time
                )
                msg = record_eval_msg()
                msg.ee_pose = query_pose.flatten().tolist()
                msg.norm_openness = query_openness
                msg.e2h_cam = e2h_frame["bgr"].flatten().tolist()
                msg.eih_cam = eih_frame["bgr"].flatten().tolist()
                msg.timestep = e2h_frame["timestep"]
                state_publisher.publish(msg)

            # 任务结束条件判断
            if vla_status == NodeStatus.IN_PROGRESS:
                if end_time - start_time > 10:
                    x, y, z = ee_pose[:3, 3]
                    if x < 0.25:
                        set_status(NodeStatus.DONE)
                        is_first = True
            elif vla_status == NodeStatus.IN_PROGRESS_STAGE0:
                if end_time - start_time > 15:
                    set_status(NodeStatus.DONE)
                    is_first = True
                else:
                    if end_time - start_time > 10:
                        x, y, z = ee_pose[:3, 3]
                        if x < 0.25:
                            set_status(NodeStatus.DONE)
                            is_first = True
            elif vla_status == NodeStatus.IN_PROGRESS_STAGE1:
                if end_time - start_time > 15:
                    set_status(NodeStatus.DONE)
                    is_first = True
                else:
                    if end_time - start_time > 10:
                        x, y, z = ee_pose[:3, 3]
                        if x < 0.28:
                            set_status(NodeStatus.DONE)
                            is_first = True
            
            rate.sleep()
    except Exception as e:
        print(f"[ERROR] Exception in record_fn: {e}")
    finally:
        pygame.quit()
        print("[INFO] Exiting recording process")









def main(head_img_shared, head_img_time_shared, right_img_shared, ee_pose_shared, ee_pose_time_shared, gripper_shared, is_head_recived, is_righthand_recived, is_eepose_recived):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-s", "--save", type=str, default="", help="path to the mp4 file")
    parser.add_argument("-x", "--xspeed", type=float, default=1.0, help="speed up video")
    opt = parser.parse_args()
    
    plan_config = PlanConfig()
    ctrl_state = Value("i", 1)  # initial as blocked
    # 0: running
    # 1: blocking
    # 2: quit
    
    record_fn(ctrl_state, plan_config, is_head_recived, is_righthand_recived, is_eepose_recived, head_img_shared, head_img_time_shared, right_img_shared, ee_pose_shared, ee_pose_time_shared, gripper_shared,state_publisher)

    print("[INFO] All finished.")



def righthand_image_callback(msg, right_img_shared, is_righthand_recived):
    #  如果是compressed的图像
    np_arr = np.frombuffer(msg.data, np.uint8)
    right_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if right_img is None:
        rospy.logwarn("Failed to decode right hand image.")
        return
    # 如果是原始图像
    # np_arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    # right_img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    np.copyto(right_img_shared, right_img)
    is_righthand_recived.value = 1
    

    

def head_image_callback(msg, head_img_shared, head_img_time_shared, is_head_recived):
    # 如果是conpressed图像
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    head_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if head_img is None:
        rospy.logwarn("Failed to decode head image.")
        return
    # 如果是原始图像
    # np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    # head_img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    
    np.copyto(head_img_shared, head_img)
    head_img_time_shared.value = msg.header.stamp.to_sec()
    is_head_recived.value = 1
    
    np.set_printoptions(suppress=True,precision=15)
    robot_time = head_img_time_shared.value
    my_time = rospy.Time.now().to_sec()
    if abs(robot_time - my_time) > 0.1:
        rospy.logwarn_throttle(1.0, "时间未对齐")
    else:
        rospy.logdebug("时间已对齐")
        
    
    
def ee_pose_callback(msg, ee_pose_shared, gripper_shared, ee_pose_time_shared, is_eepose_recived):

    quaternion = msg.rightTcpPose[:4]
    x,y,z = msg.rightTcpPose[4:7]
    base_tcp_T = np.eye(4)
    quat_norm = np.linalg.norm(quaternion)
    if quat_norm < 1e-8:
        rospy.logwarn("Received invalid zero quaternion.")
        return
    quaternion = quaternion / quat_norm
    base_tcp_T[:3, :3] = quaternion_to_matrix(quaternion)
    base_tcp_T[:3, 3] = [x, y, z]
    
    ee_pose=np.asarray(base_tcp_T)
    np.copyto(ee_pose_shared, ee_pose)
    #gripper=float(msg.right_gripper_info[0])
    gripper= 0.0
    gripper_shared.value = gripper
    ee_pose_time_shared.value = msg.header.stamp.to_sec()

    is_eepose_recived.value = 1
    rospy.logdebug(f"Received ee pose: {x}, {y}, {z}")

if __name__ == "__main__":
    
    signal.signal(signal.SIGINT, shutdown_handler)


    # RawArray 用于创建一个可以被多个进程安全读写的共享内存数组（比如用于图像数据、位姿矩阵等）。通过 RawArray 创建的共享内存区域可以在不同进程之间直接访问和修改，而不需要进行复杂的数据传输或复制。
    # vlaue 用于创建了一个可以被多个进程安全读写的浮点数（比如用于进程间通信、同步状态等）。例如，多个进程可以通过这个 Value 实例来共享和更新一个数值（如时间戳、状态标志等），而不会出现数据竞争问题。

    # 分别定义三个共享内存区域：头部相机图像、右手相机图像、末端执行器位姿、
    # 夹爪状态标记为共享变量
    
    head_img = np.zeros([480,640,3],dtype=np.uint8)
    head_img_shared = RawArray(ctypes.c_ubyte, head_img.size)
    head_img_shared = np.ctypeslib.as_array(head_img_shared).reshape(head_img.shape)
    
    
    right_img= np.zeros([480,640,3],dtype=np.uint8)
    right_img_shared = RawArray(ctypes.c_ubyte, right_img.size)
    right_img_shared = np.ctypeslib.as_array(right_img_shared).reshape(right_img.shape)
    
    ee_pose= np.zeros([4,4],dtype=np.float64)
    ee_pose_shared = RawArray(ctypes.c_double, ee_pose.size)
    ee_pose_shared = np.ctypeslib.as_array(ee_pose_shared).reshape(ee_pose.shape)
    
    gripper_shared= Value('d', 0.0)

    # 定义两个共享的时间状态变量 
    head_img_time_shared= Value('d', 0.0) # 只用一个图像时间戳的原因是ros里多个相机统一管理的 时间戳差距不大
    ee_pose_time_shared= Value('d', 0.0)
    

    # 定义三个共享的状态，表明头部相机、右手相机、末端状态是否收到
    is_head_recived = Value('i', 0)
    is_righthand_recived = Value('i', 0)
    is_eepose_recived = Value('i', 0)
    

    
    rospy.init_node('record_node', anonymous=True)



    # ================== 下面是接收机器人的状态信息 ========================
    # # 如果是压缩的图像
    # # 头部相机
    rospy.Subscriber('/headCamera/color/image_raw/compressed', CompressedImage, partial(head_image_callback, head_img_shared = head_img_shared, head_img_time_shared = head_img_time_shared, is_head_recived = is_head_recived))
    # 右手相机
    rospy.Subscriber('/rightHandCamera/color/image_raw/compressed', CompressedImage, partial(righthand_image_callback, right_img_shared = right_img_shared, is_righthand_recived = is_righthand_recived))
    
    # 机器人状态和夹爪信息
    rospy.Subscriber('/lingkun_state', wbc_state, partial(ee_pose_callback, ee_pose_shared = ee_pose_shared, gripper_shared = gripper_shared, ee_pose_time_shared = ee_pose_time_shared, is_eepose_recived = is_eepose_recived))

    
    # ==================== 下面是根据收到的字符串命令（如 'control'、'stop'、'reset' 等）来修改全局变量 vla_status；并通过 publish_current_status() 广播当前状态================
    # 当前状态广播，用在publish_current_status()
    status_pub = rospy.Publisher(
        '/vla_status',
        String,
        queue_size=10
    )

    # 订阅外部发送的/vla_state_sub状态；如 'control'、'stop'、'reset' 等；来修改全局变量 vla_status
    # 注意 vla_status 是全局变量 任何位置主要修改 就会发动不一样的东西；这就导致模型结束以后可以通过改变 vla_status  变成done
    rospy.Subscriber('/vla_state_sub', String, vla_state_sub_callback, queue_size=10)
    
    
    # 保留服务，但只做状态查询（忽略命令参数） 外部可以根据这个订阅来判断状态 他也是根据全局变量vla_status 输出状态 感觉可以去掉了 TODO
    rospy.Service('/vla_state_control', stateMachine, handle_state_control_service)


    # 在 record_fn 函数内，每次采集到新的数据后，都会通过 state_publisher.publish(msg) 向 /record_eval 话题发布消息。作用：用于将采集到的评估数据（如末端位姿、相机图像等）实时发布到 /record_eval，供其他 ROS 节点订阅和处理
    state_publisher = rospy.Publisher('/record_eval', record_eval_msg, queue_size=1)


    main(head_img_shared, head_img_time_shared, right_img_shared, ee_pose_shared, ee_pose_time_shared, gripper_shared, is_head_recived, is_righthand_recived, is_eepose_recived)
