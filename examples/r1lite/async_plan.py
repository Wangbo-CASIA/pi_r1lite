from dataclasses import dataclass
from multiprocessing import Value
import queue as queue1
from queue import Empty

from service1 import TrajPlanner
import numpy as np
import rospy
import time
import threading
from vla_msgs.msg import record_eval_msg,plan_eval_msg
from functools import partial
from enum import Enum
from std_msgs.msg import String
#1接收数据 
#2模型推理
# cmd_data_queue = manager.Queue(maxsize=10)


class NodeStatus(Enum):
    IDLE = "idle"             # 空闲状态
    IN_PROGRESS = "in_progress"  # 任务正在进行
    IN_PROGRESS_STAGE0 = "in_progress_stage0"  # 任务正在进行
    IN_PROGRESS_STAGE1 = "in_progress_stage1"  # 任务正在进行
    DONE = "done"             # 任务完成
    ERROR = "error"           # 错误状态
    

vla_state_control_srv = None
num_infer = -1

def custom_queue_put(q: queue1.Queue, obj):
    if q.full():
        try:
            _ = q.get_nowait()   
        except Empty:
            pass
        else:
            print("[WARN] Discard 1 element in queue!!!")
    q.put(obj)

@dataclass
class PlanConfig(object):

    ckpt_path1: str = "/home/chenanzhe/ssd/wangbo/robotv3/se3_va/checkpoints/collect_cup/m1129sweep1v_robotv3__0321_coffee_ros_bigdinov2_100_oneimage/ckpt400000.pt"
    ckpt_path2: str = "/home/chenanzhe/ssd/wangbo/robotv3/se3_va/checkpoints/collect_cup/m1129sweep1v_robotv3__0327_coffee_stage2_ros_bigdinov2_100_oneimage/ckpt250000.pt"
    # 相机配置
    e2h_camera_index: int = 2  # set to -1 to disable
    e2h_resolution: tuple = (640, 480)
    e2h_extr_path: str = "cam_bcT_top_1129.txt"
    eih_camera_index: int = 1  # set to -1 to disable
    eih_resolution: tuple = (640, 480)
    eih_extr_path: str = "cam_ecT_D435_1120.txt"
    # 集成配置
    ensemble: int = 20
    record_dt: float = 0.1 
    # 额外的 TODO
    # plan_hz: float = 10.0
    # min_obs_frames: int = 5



# def status_callback(msg):
#     global current_status
#     try:
#         status = NodeStatus(msg.data)
#     except ValueError:
#         rospy.logwarn(f"[Planner] Unknown status: {msg.data}")
#         return

#     with status_lock:
#         current_status = status

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

def get_current_status():
    with status_lock:
        return current_status


# 原始版本
def record_eval_callback(msg, is_head_recived, queue_planner1, queue_planner2):
    
    # global cmd_data_queue
    
    tmp_dict = {}
    tmp_dict["e2h_cam"] = {}
    tmp_dict["eih_cam"] = {}

    tmp_dict["ee_pose"] = np.array(msg.ee_pose, dtype=np.float32).reshape(4,4)
    tmp_dict["norm_openness"] = msg.norm_openness
    tmp_dict["e2h_cam"]["bgr"] = np.frombuffer(msg.e2h_cam, dtype=np.uint8).reshape(480,640,3)
    tmp_dict["eih_cam"]["bgr"] = np.frombuffer(msg.eih_cam, dtype=np.uint8).reshape(480,640,3)
    tmp_dict["e2h_cam"]["timestep"] = msg.timestep
    tmp_dict["eih_cam"]["timestep"] = msg.timestep
    
    print(tmp_dict["ee_pose"] )
    print(tmp_dict["norm_openness"])
    

    response = query_until_valid("query")
        
    
    if response.message == NodeStatus.IN_PROGRESS.value:
        custom_queue_put(queue_planner1, ("add_obs_frame", tmp_dict))
    elif response.message == NodeStatus.IN_PROGRESS_STAGE1.value:
        custom_queue_put(queue_planner2, ("add_obs_frame", tmp_dict))
    is_head_recived.value = 1



# def record_eval_callback(msg, is_head_recived, queue_planner1, queue_planner2):
#     status = get_current_status()
#     if status not in (NodeStatus.IN_PROGRESS, NodeStatus.IN_PROGRESS_STAGE1):
#         return

#     tmp_dict = {}
#     tmp_dict["e2h_cam"] = {}
#     tmp_dict["eih_cam"] = {}

#     tmp_dict["ee_pose"] = np.array(msg.ee_pose, dtype=np.float32).reshape(4,4)
#     tmp_dict["norm_openness"] = msg.norm_openness
#     tmp_dict["e2h_cam"]["bgr"] = np.asarray(msg.e2h_cam, dtype=np.uint8).reshape(480,640,3)
#     tmp_dict["eih_cam"]["bgr"] = np.asarray(msg.eih_cam, dtype=np.uint8).reshape(480,640,3)
#     tmp_dict["e2h_cam"]["timestep"] = msg.timestep
#     tmp_dict["eih_cam"]["timestep"] = msg.timestep

#     if status == NodeStatus.IN_PROGRESS:
#         custom_queue_put(queue_planner1, ("add_obs_frame", tmp_dict))
#     elif status == NodeStatus.IN_PROGRESS_STAGE1:
#         custom_queue_put(queue_planner2, ("add_obs_frame", tmp_dict))
#     is_head_recived.value = 1

def center_crop(img: np.ndarray, out_h: int, out_w: int):
    h, w = img.shape[:2]
    
    if out_h >= 0:
        start_i = h//2 - out_h//2
    else:
        out_h = h
        start_i = 0
    
    if out_w >= 0:
        start_j = w//2 - out_w//2
    else:
        out_w = w
        start_j = 0
    
    return img[start_i:start_i+out_h, start_j:start_j+out_w]


def strengthen_openness(o: np.ndarray, thresh=0.75):
    o[o < thresh] = 0.4
    o[o > thresh] = 0.8
    return o

# ==================== 模型预热函数 ====================
def warmup_planner(planner, planner_lock, num_frames=5):
    """
    使用 dummy 数据预热模型，触发 GPU 编译和内存分配。
    预热完成后重置模型，清除 dummy 数据的影响。
    """
    dummy_obs = {
        "e2h_cam": {
            "bgr": np.zeros((480, 640, 3), dtype=np.uint8),
            "timestep": 0.0
        },
        "eih_cam": {
            "bgr": np.zeros((480, 640, 3), dtype=np.uint8),
            "timestep": 0.0
        },
        "ee_pose": np.eye(4, dtype=np.float32),
        "norm_openness": 0.5
    }

    print(f"[WARMUP] Warming up planner with {num_frames} dummy frames...")
    with planner_lock:
        # 添加多帧dummy数据
        for _ in range(num_frames):
            planner.add_obs_frame(dummy_obs)
        # 执行一次推理，触发所有 lazy 操作
        _ = planner.get_action()
        # 重置模型，清空内部状态
        planner.reset()
    print("[WARMUP] Warmup completed, planner reset to clean state.")


def clear_queue(q: queue1.Queue):
    with q.mutex:
        q.queue.clear()





def plan_fn(
    plan_config: PlanConfig, 
    is_head_recived,
    queue_planner1,   # 新增第一个模型专用队列
    queue_planner2    # 新增第二个模型专用队列
):
    # 发送节点
    global vla_state_control_srv,num_infer

    data_sub = rospy.Subscriber(
        '/record_eval',
        record_eval_msg,
        partial(
            record_eval_callback,
            is_head_recived=is_head_recived,
            queue_planner1=queue_planner1,
            queue_planner2=queue_planner2
        ),
        queue_size=1
    )

    publisher_result = rospy.Publisher('/plan_eval', plan_eval_msg, queue_size=1)
    # obs_state = {
    #     "lock": threading.Lock(),
    #     "processed": {1: 0, 2: 0},
    #     "last_infer": {1: 0, 2: 0},
    # }
    
    planner1 = TrajPlanner(
        ckpt_path=plan_config.ckpt_path1,
        device="cuda:0",
        ensemble=plan_config.ensemble
    )

    planner2 = TrajPlanner(
        ckpt_path=plan_config.ckpt_path2,
        device="cuda:0",
        ensemble=plan_config.ensemble
    )

    # 两个细粒度的GPU锁，分别保护两个模型的推理过程，避免交叉干扰
    planner1_lock = threading.Lock()
    planner2_lock = threading.Lock()

    warmup_planner(planner1, planner1_lock, num_frames=5)
    warmup_planner(planner2, planner2_lock, num_frames=5)


    while not rospy.is_shutdown():
        if is_head_recived.value: 
            break
        else:
            print("[INFO] Waiting for record msg from ros...")
            time.sleep(0.5)

    

    # status_sub = rospy.Subscriber('/vla_status', String, status_callback, queue_size=10)
    
    def execute_cmd_bg(planner, queue, planner_lock):
        while not rospy.is_shutdown():
            try:
                cmd_data = queue.get(timeout=0.1)
                cmd = cmd_data[0]
                data = cmd_data[1:]
                # 加锁执行涉及 GPU 的操作
                with planner_lock:
                    getattr(planner, cmd)(*data)
                # if cmd == "add_obs_frame":
                #     mark_obs_processed(obs_state, planner_id)
            except Empty:
                continue
            except Exception as e:
                rospy.logerr(f"[Planner] Failed to execute {cmd}: {e}")
	            
    bg_thread1 = threading.Thread(target=execute_cmd_bg, args=(planner1, queue_planner1, planner1_lock), daemon=True)
    bg_thread2 = threading.Thread(target=execute_cmd_bg, args=(planner2, queue_planner2, planner2_lock), daemon=True)
    bg_thread1.start()
    bg_thread2.start()

   



    while not rospy.is_shutdown():

       #status = get_current_status()

        response = query_until_valid("query")

        if reponse.message in (NodeStatus.DONE, NodeStatus.IDLE, NodeStatus.IN_PROGRESS_STAGE0):
            if status != last_reset_status:
                print(f"detect reset condition ({status.value}), resetting both planners")
                num_infer = -1
                
                with planner1_lock:
                    planner1.reset()
                with planner2_lock:
                    planner2.reset()
                    
                clear_queue(queue_planner1)
                clear_queue(queue_planner2)
                # reset_obs_state(obs_state)
                # last_reset_status = status
            # rate.sleep()
            continue
        # last_reset_status = None
	        
         # 推理阶段
        t0 = time.perf_counter()
        action = None


        if response.message == NodeStatus.IN_PROGRESS.value:
            with planner1_lock:
                action = planner1.get_action()
            print("model 1 inference")
        elif response.message == NodeStatus.IN_PROGRESS_STAGE1.value:
            with planner2_lock:
                action = planner2.get_action()
                print("model 2 inference")
        
        if action is None:
            print("action none----------------")
            time.sleep(0.1)
            continue
                

        msg = plan_eval_msg()
        msg.ee_pose = action[0].flatten().tolist() # (16,4,4)
        msg.norm_openness = action[1]
        msg.timesteps = action[2].flatten().tolist()


        num_infer += 1
        if num_infer>1:
            publisher_result.publish(msg)

        
        t1 = time.perf_counter()
        print("[INFO] Planning iteration dt: {:.3f}s".format(t1 - t0))
        rate.sleep()
    

    bg_thread1.join()
    bg_thread2.join()
    print("[INFO] Exiting planning process")




if __name__=="__main__":
    rospy.init_node('inference_node', anonymous=True)
    plan_config = PlanConfig()
    is_head_recived = Value('i', 0)
    
     # 创建两个独立队列（线程安全，容量根据需要调整）
    queue_planner1 = queue1.Queue(maxsize=10)
    queue_planner2 = queue1.Queue(maxsize=10)

    rospy.wait_for_service('/vla_state_control')
    vla_state_control_srv = rospy.ServiceProxy('/vla_state_control', stateMachine)

    plan_fn(plan_config,is_head_recived,queue_planner1, queue_planner2)
     
