import time
import threading
import numpy as np
from multiprocessing import Value
from fast_msgs.srv import stateMachine
from teleop_rec4.post_process import interp_SE3_sep, interp_linear
from scipy.spatial.transform import Rotation
import rospy
from sensor_msgs.msg import JointState
from fast_msgs.msg import armer
from functools import partial
from vla_msgs.msg import plan_eval_msg
from enum import Enum

import signal
import sys

def shutdown_handler(sig, frame):
    rospy.signal_shutdown("Received SIGINT")
    sys.exit(0)



current_state = {}
state_lock = threading.Lock()

class NodeStatus(Enum):
    IDLE = "idle"             # 空闲状态
    IN_PROGRESS = "in_progress"  # 任务正在进行
    IN_PROGRESS_STAGE0 = "in_progress_stage0"  # 任务正在进行
    IN_PROGRESS_STAGE1 = "in_progress_stage1"  # 任务正在进行
    DONE = "done"             # 任务完成
    ERROR = "error"           # 错误状态
#vla_status = NodeStatus.IDLE    
vla_state_control_srv = None

current_status = NodeStatus.IDLE

def status_callback(msg):
    global current_status

    status_str = msg.data

    try:
        current_status = NodeStatus(status_str)
        rospy.loginfo(f"[Exec] Receive status: {current_status.value}")

    except ValueError:
        rospy.logwarn(f"[Exec] Unknown status: {status_str}")

    
def action_interp(
    train_poses,
    train_openness, 
    train_timesteps,
    query_time
):

    if query_time >= train_timesteps[-1]:
        query_pose = train_poses[-1]
        query_openness = train_openness[-1]
        print("最后一个动作")
    elif query_time <= train_timesteps[0]:
        query_pose = train_poses[0]
        query_openness = train_openness[0]
        print("第一个动作")
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
        print(f"第{l_index}个动作")
    print(query_pose)
    return query_pose, query_openness

def quaternion_to_matrix(q):
    """使用SciPy将四元数转换为旋转矩阵"""
    # SciPy的四元数格式是[x, y, z, w]
    r = Rotation.from_quat([q[0], q[1], q[2], q[3]])
    return r.as_matrix()



def plan_callback(msg,is_plan_recived):
    
    global current_state
    next_state = {
        "base_ee_Ts": np.array(msg.ee_pose, dtype=np.float32).reshape(16,4,4),
        "norm_openness": np.array(msg.norm_openness, dtype=np.float32),
        "timesteps": np.array(msg.timesteps, dtype=np.float64)
    }
    with state_lock:
        current_state = next_state
    
    is_plan_recived.value = 1
    
def query_until_valid(command, max_attempts=100, delay=0.01):
    for _ in range(max_attempts):
        try:
            resp = vla_state_control_srv(command, "none", "none", [])
            if resp.message != "Unknown command":
                return resp
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call failed: {e}")
        rospy.sleep(delay)
    raise RuntimeError("No valid response after retries")


def exec_fn(is_plan_recived, pub_right_target, pub_left_target=None, pub_body_target=None, pub_dexhand=None):
    
    
    global current_status
    global current_state
    while not rospy.is_shutdown():
        if is_plan_recived.value: # 循环检查共享变量 is_plan_recived 的值，直到其变为非零（表示已收到 /plan_eval 消息）
            break
        else:
            print("[INFO] Waiting for plan msg from ros...")
            time.sleep(0.5)


    
    rate = rospy.Rate(30)
    desired_traj = None
    current_desired_ee_pose = None
 
    while not rospy.is_shutdown():

        
        
        if current_status.value == NodeStatus.DONE.value: 
            # 如果返回的字符串等于 NodeStatus.DONE.value（通常表示状态机已完成），则打印检测信息，并将全局变量 current_state 重置为空字典
            print("detect done!!!!!!!!!!!!!!!!!!!!")
            with state_lock:
                current_state = {}
            rate.sleep()
            continue
        if current_status.value == NodeStatus.IN_PROGRESS_STAGE0.value: 
            # 如果返回的字符串等于 NodeStatus.DONE.value（通常表示状态机已完成），则打印检测信息，并将全局变量 current_state 重置为空字典
            print("detect YOLO!!!!!!!!!!!!!!!!!!!")
            with state_lock:
                current_state = {}
            rate.sleep()
            continue


        with state_lock:
            desired_traj = dict(current_state)
        if len(desired_traj) > 0:
            now = rospy.Time.now()
            traj_timesteps = desired_traj["timesteps"]
            traj_ee_poses = desired_traj["base_ee_Ts"]
            traj_openness = desired_traj["norm_openness"]

            desired_ee_pose, desired_openness = action_interp(
                train_poses=traj_ee_poses,
                train_openness=traj_openness,
                train_timesteps=traj_timesteps,
                #query_time=now.to_sec() + 1/50 * 2
                query_time=now.to_sec() + 1/30
            )
            print(f"openness:{desired_openness}")
            

    

            current_desired_ee_pose = desired_ee_pose

            right_quat = Rotation.from_matrix(desired_ee_pose[:3,:3]).as_quat()
            
  
            
            # desired_ee_pose[2,3] += 0.01
            right_pos = desired_ee_pose[:3,3]

            
            right_pos = right_pos.astype(float).tolist()
            right_quat = right_quat.astype(float).tolist()
            
            # quaternion: x, y, z, w
            msg = armer()
            msg.cartesian_cmd = right_pos + right_quat
            msg.gripper_cmd = [desired_openness]
            pub_right_target.publish(msg)
            

           
            
            left_msg = armer()
            left_pose = [0.0841, 0.1814, 0.4512]
            left_quat = [0.7097, 0.0589, -0.7019, 0.0112]
            left_msg.cartesian_cmd = left_pose + left_quat
            left_msg.gripper_cmd = [desired_openness]
            if pub_left_target is not None:
                pub_left_target.publish(left_msg)
            
            body_msg = armer()
            body_pose = [0.1473, 0.000087, 1.0509]
            body_quat = [-0.00013, 0.2588, 0.00042,0.9659]
            body_msg.cartesian_cmd = body_pose + body_quat
            body_msg.gripper_cmd = [desired_openness]
            if pub_body_target is not None:
                pub_body_target.publish(body_msg)

            

            # 发送灵巧手动作
            joint_state = JointState()
            joint_state.name = [
                'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
                'joint_6', 'joint_7', 'joint_8', 'joint_9', 'joint_10'
            ]
            joint_state.position = [0,-1.2,0.47*desired_openness,0,1.5*desired_openness,1.5*desired_openness,0,1.5*desired_openness,0,1.5*desired_openness]
            print(f'desired_openness:{desired_openness}')
            if pub_dexhand is not None:
                pub_dexhand.publish(joint_state)

        rate.sleep()
    
    print("[INFO] Exiting execution process")


def publish_ee_cmd(self, left_pos, left_quat, right_pos, right_quat, left_hand_rate, right_hand_rate, gui_control):

        left_pos = list(left_pos)
        left_quat = list(left_quat)
        right_pos = list(right_pos)
        right_quat = list(right_quat)

        # quaternion: x, y, z, w
        cmd = left_pos + left_quat + right_pos + right_quat + [left_hand_rate, right_hand_rate] + [gui_control] + [0, 0, 0] + [0, 0, 0, -1]

        self.ee_target_msg.cmdVector = cmd
        self.ee_target_msg.size = len(cmd)
        self.pub_ee_target.publish(self.ee_target_msg)



from std_msgs.msg import String
if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
   
    # 创建两个整型共享变量，初始值为0；这些共享变量可以在多个进程之间安全读写，用于状态同步。
    is_plan_recived = Value('i', 0)
    rospy.init_node('exec_node', anonymous=True) # rospy.init_node 初始化ROS节点，节点名为 exec_node
    rospy.wait_for_service('/vla_state_control') # 阻塞当前线程，直到名为 /vla_state_control 的 ROS 服务被成功发布（即服务端已准备就绪）
    vla_state_control_srv = rospy.ServiceProxy('/vla_state_control', stateMachine) 
    
    # 改为订阅状态topic，不再疯狂轮询service
    rospy.Subscriber(
        '/vla_status',
        String,
        status_callback,
        queue_size=10
    )

    # 接收模型输出
    plan_sub = rospy.Subscriber('/plan_eval', plan_eval_msg, partial(plan_callback, is_plan_recived = is_plan_recived))
    # 发送给机器人
    pub_right_target = rospy.Publisher('/pyroki_right_cmd', armer, queue_size=1)
    pub_left_target = rospy.Publisher('/pyroki_left_cmd', armer, queue_size=1)
    pub_body_target = rospy.Publisher('/pyroki_body_cmd', armer, queue_size=1)
    pub_dexhand = rospy.Publisher('/right_manus_hand_cmd', JointState, queue_size=1)
    
    
    exec_fn(is_plan_recived, pub_right_target, pub_left_target, pub_body_target, pub_dexhand)
 
