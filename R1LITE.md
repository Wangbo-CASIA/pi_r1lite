#### 一、 将准备好Raw数据转出lerobot格式

脚本位于 examples/r1lite/1convert_r1lite_raw_to_lerobot_abs_joint.py

需要额外配置其中的

default_experiment="r1lite_pack_phone_new_state", 每个任务对应一个独立实验目录，每个实验维护自己的 `config.yaml`
default_action_space="abs_joint" 

###### 注意：实验配置位于 experiments/r1lite/“r1lite_pack_phone_new_state” 需要和default_experiment对应上

一个实验配置大致分成这些部分：

```text
experiment
  name              实验 id，也是 --experiment 使用的名字
  task_desc         写入 LeRobot 数据集的任务文本
  prompt            rollout 时发送给 policy server 的 VLA prompt

robot 机器人端侧配置
  server_url        机器人 server 地址，12.12 机器人通常是 http://192.168.12.12:8001
  record_output_dir 机器人本机 rosbag 输出父目录

local 当前训练机配置
  raw_dir           推理机本地 RAW 父目录，converter 从这里读取
  lerobot_root      推理机本地 LeRobot 数据集父目录

rollout
  policy_host       policy server host
  policy_port       policy server port
  control_hz        控制频率
  actions_per_infer 每次推理后连续执行的 action 数
  max_steps         单次 rollout 最大步数
  timeout           HTTP 请求超时
  rtc               RTC 推理配置，可在 YAML 中默认启用或关闭
  gripper_*         gripper open/close 取值

hg_dagger
  intervention_arm  SpaceMouse 控制 left/right/dual
  deadzone          SpaceMouse 摇杆死区
  gripper_threshold gripper 按键阈值
  teleop_*_scale    人类介入时的物理控制尺度

data
  fps               LeRobot 数据集 fps
  raw_dir_glob      converter 搜索 RAW episode 的 glob
  recursive         是否递归搜索 RAW episode
  gripper_threshold RAW gripper 值二值化阈值

action_spaces
  joint_delta       关节相对控制数据、训练和 rollout 配置
  abs_eef           末端绝对位姿控制数据、训练和 rollout 配置
  delta_eef         末端相对位姿控制数据、训练和 rollout 配置
```

至少需要修改：注意 `train_config` 

```text
experiment.name
experiment.task_desc
experiment.prompt
local.raw_dir
action_spaces.<space>.repo_id
action_spaces.<space>.lerobot_dir
action_spaces.<space>.train_config  `train_config` 名称，还需要在 `src/openpi/training/config.py` 中增加或确认对应 OpenPI 训练配置，并让它指向 YAML 里对应的 `repo_id`。如果只是复用同一套 state/action schema，可以复用已有训练配置的结构，但 repo_id、dataset root 和 norm-stat asset 必须和新实验一致。
```

#### 二、 配置训练所使用的 train_config LeRobotR1LiteDataConfig r1lite_policy.py

- src/openpi/training/config.py 多了一个
TrainConfig(name="pi05_r1lite_pack_phone_new_state_joint_lora",

- 又多加了一个 data=LeRobotR1LiteDataConfig 类 定义了模型输入到输出的管道；其中调用了r1lite_policy

- 针对不同的动作表征方式，在 openpi/src/openpi/policies/ 多了一个 r1lite_policy.py 的脚本 定义了训练模型的输入和输出格式

具体的细节看数据处理和模型输入.md


#### 三、 求lerobot格式数据的Norm Stats 

直接使用原始的 scripts/compute_norm_stats.py

- `HF_LEROBOT_HOME` 要指向推理机上的 LeRobot 数据集父目录。不要让 `LEROBOT_HOME` 指向旧目录，否则 `compute_norm_stats.py` 和训练可能读取错数据集。

```bash
HF_LEROBOT_HOME=/home/robot/wangbo/project/VLA-RL/conrft-r1lite/data/lerobot_openpi \
uv run scripts/compute_norm_stats.py \
  --config-name pi05_r1lite_pack_phone_new_state_abs_joint_lora
```



#### 四、模型训练

OpenPI 仍然使用 pi0 LoRA 训练链路：

```text
base checkpoint: gs://openpi-assets/checkpoints/pi0_base/params
LoRA variants: gemma_2b_lora + gemma_300m_lora
model action_dim: 32
```

R1Lite 的真实 action 维度不是 32。模型 action_dim 保持 32 是为了兼容 `pi0_base` checkpoint；真实 state/action 会在 pipeline 中通过 `PadStatesAndActions` 补到 32 维，推理输出再裁回各控制方式自己的真实 action 维度。


```bash
HF_LEROBOT_HOME=/home/ps/VLA-RL/conrft-r1lite/data/lerobot_openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi05_r1lite_pack_phone_new_state_abs_joint_lora \
  --exp-name=<run_name> \
  --overwrite
```
HF_LEROBOT_HOME=/home/robot/wangbo/project/VLA_own/data/r1lite_pack_phone_0707 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py r1lite_pack_phone_abs_joint_crop_head_image_0707 --exp-name=r1lite_pack_phone_abs_joint_crop_head_image_0707 --overwrite


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_r1lite_pack_phone_new_state_abs_joint_lora --exp-name=pi05_r1lite_pack_phone_new_state_abs_joint_lora --overwrite

`<train_config>` 必须和 `src/openpi/training/config.py` 中的配置名一致。新任务如果新增 repo_id 但忘记更新训练配置，会在数据集加载或 norm stats 计算阶段直接报错。


模型会保存在 checkpoints/pi05_r1lite_pack_phone_new_state_abs_joint_lora 中



#### 五、模型推理

1）离线开环 Action 可视化

训练完成后用 checkpoint 启动 policy server：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_r1lite_pack_phone_new_state_abs_joint_lora \
  --policy.dir=/home/robot/wangbo/project/VLA_own/openpi/checkpoints/pi05_r1lite_pack_phone_new_state_abs_joint_lora/pi05_r1lite_pack_phone_new_state_abs_joint_lora/14999
```


```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=r1lite_pack_phone_abs_joint_crop_head_image_0707 \
  --policy.dir=/home/robot/wangbo/project/VLA_own/openpi/checkpoints/r1lite_pack_phone_abs_joint_crop_head_image_0707/r1lite_pack_phone_abs_joint_crop_head_image_0707/14999
```

这条命令没有传 --port，所以 policy server 默认是：
0.0.0.0:8000

再启动 Web 可视化页面：

```bash
uv run examples/r1lite/visualize_r1lite_open_loop_actions.py \
  --experiment r1lite_pack_phone_new_state \
  --action-space abs_joint \
  --host 127.0.0.1 \
  --port 7861
```

然后打开：

```text
http://127.0.0.1:7861

```

整个流程是：

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py policy:checkpoint   --policy.config=r1lite_pack_phone_abs_joint_crop_head_image_0707  --policy.dir=./checkpoints/r1lite_pack_phone_abs_joint_crop_head_image_0707/r1lite_pack_phone_abs_joint_crop_head_image_0707/14999


### 1）启动 Policy Server   scripts/serve_policy.py 


功能1：找到模型，加载模型参数；加载数据处理流程

- 先通过配置名找到 TrainConfig，再从 checkpoint 目录恢复模型；

1. 首先确认模型，位于train_config中的

model=pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

所以这个 policy 的性质是：

模型: pi0.5
动作 horizon: 每次推理输出 10 步 action
状态输入: discrete_state_input=True，会把 state 也放进 prompt/token 相关输入
LoRA: paligemma 和 action expert 都是 lora variant
动作空间: abs_joint，但代码实际复用了 R1LiteInputs/R1LiteOutputs

2. 加载模型参数：检查检查 checkpoint；如果有model.safetensors，按 PyTorch 模型加载，否则按 JAX checkpoint 加载 params。


3. 从 checkpoint/assets 或 data config 中拿 normalization stats。
组装一个 Policy 对象，里面包含模型、输入 transform、输出 transform。


server 不是裸模型推理，而是“模型 + 数据预处理 + 反归一化 + 输出裁剪”的完整 policy。


- 功能2：Server 的推理服务协议： websocket server 实现在 src/openpi/serving/websocket_policy_server.py

加载完 policy 后，serve_policy.py 会启动 websocket server，默认端口是 8000；你的命令没有传 --port，所以 policy server 默认是：0.0.0.0:8000 （7861 是后面 Web 可视化页面的端口。）

- 客户端发一个 observation，格式是 msgpack_numpy。
- 当前server 调用 policy.infer(obs)。
- server 返回 action dict，里面一般有 
actions 其中actions shape ~= [action_horizon, action_dim] action_horizon=10，R1Lite joint 输出最后会裁成 14 维，所以典型输出是：[10, 14]
policy_timing 模型核心推理时间
server_timing 整个policy.infer 整体时间，包括包括输入的transform、模型推理、 输出 transform 反归一化


policy.infer()函数位于 openpi/src/openpi/policies/policy.py 主要流程就是
 -> input transforms R1Lite 输入变成模型输入
 -> 加 batch 维度
 -> JAX/PyTorch tensor
 -> Observation.from_dict()
 -> model.sample_actions()
 -> 去 batch 维度
 -> output transforms 转回 R1Lite action
 -> 返回 actions
 所以你的 policy server 收到的是单帧 observation，但输出的是一个 action chunk

  def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
      inputs = jax.tree.map(lambda x: x, obs)
      inputs = self._input_transform(inputs)
      ...
      observation = _model.Observation.from_dict(inputs)
      outputs = {
          "state": inputs["state"],
          "actions": self._sample_actions(...)
      }
      outputs = self._output_transform(outputs)
      return outputs

其中：
R1Lite的输入输出

输入的 14 维来自 r1lite_policy.py  左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1 = 14；在模型输入前补零到 32 维。模型输出 actions: [action_horizon, 32] 最终给机器人/可视化页面用的是前 14 维  {"actions": np.asarray(data["actions"][:, :14])}

### 2）启动 Web 可视化页面   visualize_r1lite_open_loop_actions.py


1. 读取 experiments/r1lite/<experiment>/config.yaml 实验yaml

读取 rollout 配置
rollout:
  policy_host: localhost
  policy_port: 8000
它推理时会连接：ws://localhost:8000


2. 启动 Flask 网页服务

-host 127.0.0.1 --port 7861 的含义：
127.0.0.1：仅本机可以访问，不对局域网暴露。
7861：Flask HTTP 服务端口。
浏览器访问：http://127.0.0.1:7861。


3. 根据 config 找 LeRobot dataset 位置

某个时刻 t 的样本大致包括：（lerobot的数据格式）
  {
      "observation.images.head": ...,
      "observation.images.left_wrist": ...,
      "observation.images.right_wrist": ...,
      "observation.state": ...,
      "action": ...,
      "task": ...,
  }

转成 （发给 policy server 的运行时 observation）

{
  "images": {
    "head": ...,
    "left_wrist": ...,
    "right_wrist": ...,
  },
  "state": ...,
  "prompt": ...
}


4. 请求到达 server 后，最终调用：
[Policy.infer()] 进行推理

R1Lite 原始观测
  -> 输入 transform
  -> 归一化 / 图像预处理 / prompt token 化
  -> state padding
  -> Pi0.5 模型采样 action chunk
  -> 反归一化
  -> R1Lite 输出格式转换
  -> 返回 actions
所以单次模型推理通常生成：[10, 32]


逐段跑 open-loop 推理，设定actions_per_infer: 5

时刻 0 的观测 -> 预测 action[0:5]  -> 与 GT action[0:5] 比较
时刻 5 的观测 -> 预测 action[0:5]  -> 与 GT action[5:10] 比较
时刻 10 的观测 -> 预测 action[0:5] -> 与 GT action[10:15] 比较

再把预测动作和 GT 动作画出来

开环的关键在于：模型只基于开始时刻 t 的观测预测未来，不会在预测中途重新读取后续真实图像和状态。

所以此页面适合先检查 checkpoint 的动作趋势、维度、归一化和时序。它能发现很多问题，但不能完全证明真实执行一定成功，因为接触、时延、视觉变化和执行误差都还没有进入闭环。



## 普通 Rollout

先试跑/预演模式。用来检查参数、动作格式、payload 是否合理，比直接让机器人动安全很多 
主要检查 policy server、机器人 server、观测和 action shape：




```bash
uv run scripts/run_r1lite_openpi_delta_eef_policy.py \
  --experiment <experiment> \
  --debug \
  --max-steps 10
```

真实执行时加 `--execute`：

```bash
uv run scripts/run_r1lite_openpi_delta_eef_policy.py \
  --experiment <experiment> \
  --execute \
  --max-steps 120
```

不同 action space 使用不同 rollout 脚本：

```text
joint_delta -> scripts/run_r1lite_openpi_policy.py
abs_eef     -> scripts/run_r1lite_openpi_abs_eef_policy.py
delta_eef   -> scripts/run_r1lite_openpi_delta_eef_policy.py
```

--experiment <experiment> 会去读： experiments/r1lite/<experiment>/config.yaml

比如你当前仓库里有 [experiments/r1lite/r1lite_pack_phone_new_state/config.yaml。
配置会补齐这些运行参数：

prompt； 
如果你的命令行没有传入 --prompt "pick up the phone and put it in the box" 则会从实验config文件中读取
experiment:
  prompt: "..."
如果配置里也没有，就用脚本里的 DEFAULT_PROMPT。



robot_server 是机器人控制服务的 HTTP 地址。配置文件里为
  robot:
    server_url: "http://192.168.12.12:8001"
1读取机器人当前状态，包括
三路相机图像
左右臂 joint_pos
左右夹爪 gripper_pose
状态有效性 meta.validity
2 以及在 --execute 时：
POST <robot_server>/action
把生成的动作发给机器人执行。



policy_host / policy_port 
rollout:
  policy_host: "localhost"
  policy_port: 8000
这两个是 OpenPI 模型推理服务的 websocket 地址。通常先启动的这个服务：
uv run scripts/serve_policy.py policy:checkpoint
脚本会连接：
ws://<policy_host>:<policy_port>
然后把 observation 发过去：
images + state + prompt
policy server 返回：
actions

robot_server 是“机器人在哪儿”
policy_host / policy_port 是“模型在哪儿”
run_r1lite_openpi_policy.py
        |
        | GET /state
        v
robot_server
机器人状态、图像
        |
        v
run_r1lite_openpi_policy.py
        |
        | websocket infer(images + state + prompt)
        v
policy_host:policy_port
OpenPI policy server
        |
        | actions
        v
run_r1lite_openpi_policy.py
        |
        | POST /action, 只有 --execute 才会发
        v
robot_server
机器人执行动作


control_hz
actions_per_infer
gripper 阈值和开合值
timeout
rtc 配置
hg_dagger 配置