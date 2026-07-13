#### 一、 lerobot格式数据分析

从数据的metainfo.json可以看出：

当前观测：三路 360×640 视频；换了名称
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist
当前关节状态
  observations.state.qpos              # 当前 14 维关节状态
当前绝对关节动作：
  action.qpos  # 当前 14 维关节状态
对应的任务index
  task_index


#### 二、 lerobot格式数据compute_norm_stats

    export HF_LEROBOT_HOME="/home/robot/wangbo/project/VLA_own/data/r1lite_pack_phone_0707" && .venv/bin/python scripts/compute_norm_stats.py --config-name r1lite_pack_phone_abs_joint_crop_head_image_0707


主要功能是：用与训练一致的“数据读取 + R1Lite 字段适配”过程，遍历全量训练数据，统计最终训练会归一化的 state 和 actions 的 mean/std/q01/q99，写入 norm_stats.json

主要流程是：命令行 config_name
→ 构建好数据处理流程
→ LeRobot 读取原始 sample，并构造 action.qpos 的未来 50 帧
→ 走R1LiteRepack / R1LiteInputs
→ 取batch的数据
→ 对 state 和 actions 累积统计
→ 保存 norm_stats.json


1）构建好整个数据处理规则
    TrainConfig(
            name="r1lite_pack_phone_abs_joint_crop_head_image_0707",
            ...
            data=LeRobotR1LiteDataConfig(
              
                action_space="abs_joint", # 决定了模型输入的格式
                action_sequence_keys=("action.qpos",),
                base_config=DataConfig(
                    root=".../r1lite_pack_phone_0707/pack_up_a_smart_phone/", # 数据集位置
                    prompt_from_task=True,
                ),
            ),

LeRobotR1LiteDataConfig 在创建 data_config 时带入：
repack_transforms = [R1LiteRepack()]
data_transforms = [R1LiteInputs()]

2）LeRobot 先读取原始数据，并构造动作序列
create_torch_dataloader() 调用：
dataset = _data_loader.create_torch_dataset(
    data_config, action_horizon, model_config
)

构造动作序列的意思是：
在进入 R1LiteRepack() 前，LeRobot 已经把 action.qpos 按照设定的chunk大小，组织成了：action.qpos: [50, 14]  

注意：由于lerobot默认的action 没有 action.qpos key；因此，需要在第一步骤指定LeRobotR1LiteDataConfig 里 action_sequence_keys=("action.qpos",)  如果不设置 action_sequence_keys=("action.qpos",)，流程会在 LeRobotDataset 读取阶段就因默认尝试 action 列而失败，根本到不了 R1LiteRepack。


3）走R1LiteRepack / R1LiteInputs

- 需要在R1LiteRepack 对键名进行适配 （如果其他的命名方式 需要在这里进行修正）

observation.images.cam_high →  images.head
observation.images.cam_left_wrist → images.left_wrist
observation.images.cam_right_wrist → images.right_wrist

observations.state.qpos → state
action.qpos [50,14]     → actions

- R1LiteInputs再统一改成OpenPI 的统一命名：

head        → base_0_rgb
left_wrist  → left_wrist_0_rgb
right_wrist → right_wrist_0_rgb
并确认：
state   是 14 维
actions 是 [50, 14]


再进行统计；并且只统计：机器人的真实 14 维 state、14 维 action，而不是模型内部补零后的 32 维表示。

因为模型正常训练过程中是先进行norm  再扩充到32维度的；


#### 三、模型训练

    HF_LEROBOT_HOME="/home/robot/wangbo/project/VLA_own/data/r1lite_pack_phone_0707" XLA_PYTHON_CLIENT_MEM_FRACTION="0.9" .venv/bin/python scripts/train.py r1lite_pack_phone_abs_joint_crop_head_image_0707 --exp-name r1lite_pack_phone_new_state2

第一个参数是训练的trainconfig配置名
第二个参数是模型保存位置

分析一下整个训练时的数据处理流程：

1）训练配置里强制写了 root="/home/robot/wangbo/project/VLA_own/data/r1lite_pack_phone_0707/pack_up_a_smart_phone/" LeRobotDataset 会从这个目录读取，而不是按 HF_LEROBOT_HOME/repo_id 自动拼接路径。

本地数据集元数据显示：
301 个 episode、236311 帧
15 FPS
三路 360×640 视频：cam_high、cam_left_wrist、cam_right_wrist
当前关节状态：observations.state.qpos，14 维
绝对关节动作：action.qpos，14 维

2）create_torch_dataset() 在 [data_loader.py (line 134)] 创建 LeRobot 数据集；
- 并扩充动作chunk；模型不是“给当前画面预测一帧动作”，而是学习预测长度为 50 的绝对关节动作序列
- prompt_from_task=True 需要这个 task_index 对应的 meta/tasks.jsonl 的任务映射中取出自然语言任务描述，形成 prompt对。

变成了：
当前观测：
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist
  observations.state.qpos              # 当前 14 维关节状态
  Prompt
监督目标：
  action.qpos[t : t+50]                # 未来连续 50 帧，每帧 14 维


3） 统一键名称到OPENPI

- [R1LiteRepack]将 LeRobot 原始键重命名为统一键：

observation.images.cam_high        → images.head
observation.images.cam_left_wrist  → images.left_wrist
observation.images.cam_right_wrist → images.right_wrist

observations.state.qpos            → state
action.qpos                        → actions
Prompt                        → prompt

- 随后使用 R1LiteInputs将 LeRobot 原始键重命名为统一键：

images.head        → image.base_0_rgb
images.left_wrist  → image.left_wrist_0_rgb
images.right_wrist → image.right_wrist_0_rgb
state              → state
actions            → actions
Prompt                        → prompt


4）归一化、图像和文本处理
→ Normalize：state 和 actions 会根据各自统计量被缩放到大致 [-1, 1] 范围
→ InjectDefaultPrompt：
→ ResizeImages(224, 224)：三路原始 360×640×3 图像被 resize-with-pad 为 224×224×3
→ TokenizePrompt： PaliGemma tokenizer 生成；TokenizePrompt 会把“任务文本 prompt”和“当前 state”一起编码进 tokenized_prompt 里。但更准确地说，不是生成两套 token，而是生成一条合并后的 token 序列：Task: <任务文本>, State: <离散化后的 state 数字序列>; Action:
先把 state 每一维离散到 0~255 左右:
  [23, 180, 91, ...]
拼成字符串:
  "Task: pack up a smart phone, State: 23 180 91 ...;
   Action: "
再整体送进 SentencePiece tokenizer
也就是状态被编码进了 tokenized_prompt 里
→ PadStatesAndActions(32)：Pi0.5 的统一 action_dim=32，所以最后补零为：state: 32 维  actions: 50 × 32 维


5） 16 个 已转换样本会被 torch.utils.data.DataLoader 堆叠为一个 batch；

前面每一条样本已经变成类似：
state: [32]
actions: [50, 32]
image.base_0_rgb: [224, 224, 3]
image.left_wrist_0_rgb: [224, 224, 3]
image.right_wrist_0_rgb: [224, 224, 3]
tokenized_prompt: [L]
image_mask 表示某一路相机图像在当前样本里是否有效。即这路图像 token 要不要参与 attention
image_mask.base_0_rgb: True
image_mask.left_wrist_0_rgb: True
image_mask.right_wrist_0_rgb: True
tokenized_prompt_mask: [L] 表示 token 序列里哪些位置是真实 token，哪些位置是 padding。因为 prompt token 会被 pad/truncate 到固定长度 max_token_len
在模型里二者最后都会进入 input_mask，用于构造 attention mask。图像 mask 控制图像 token 是否有效；prompt mask 控制语言/state token 是否有效。


然后 torch.utils.data.DataLoader 用 _collate_fn 做：np.stack(..., axis=0)  所以 16 条样本会变成：
image.base_0_rgb: [16, 224, 224, 3]
image.left_wrist_0_rgb: [16, 224, 224, 3]
image.right_wrist_0_rgb: [16, 224, 224, 3]
tokenized_prompt:      [16, L]
state: [16, 32]
actions: [16, 50, 32]
tokenized_prompt_mask: [16, L]
image_mask.base_0_rgb: [16]
image_mask.left_wrist_0_rgb: [16]
image_mask.right_wrist_0_rgb: [16]

然后data_loader.py中的DataLoaderImpl会把 batch 字典拆成：Observation.from_dict(batch) 和 batch["actions"]；  batch = (observation, actions)
- observation:
  images
  image_masks
  state
  tokenized_prompt
  tokenized_prompt_mask
- actions:
  [16, 50, 32]

Observation.from_dict 里还会把 uint8 图像转成 [-1, 1] 的 float 图像，此外如果训练的torch模型还需要在这里转成 NHWC → NCHW


主要注意的是
1 图像在读取lerobot前就已经转出了RGB格式；不需要再做
RGB: [R, G, B] -- >  BGR: [B, G, R] 
2 在整个处理过程中 transform 这一路通常先用 numpy 图像；即 
单张: [H, W, C]
batch 后: [N, H, W, C] = NHWC

VS Code 这个 scripts/train.py 训练，默认是 JAX 训练，不是 PyTorch 训练。
如果训练的是jax模型，则一般是 NHWC 进入 JAX 图像模型；
如果走 PyTorch framework，才会在 Observation.from_dict 里转成 NCHW
PyTorch 的卷积/视觉模型通常期望：[N, C, H, W] = NCHW  因此最终batch送到模型之前需要转成(16,3,224,224)


6）送入到模型；核心是model.compute_loss

上述整个过程为 ./scripts/train.py 下面的三个步骤

data_loader = _data_loader.create_data_loader(...)
data_iter = iter(data_loader)
batch = next(data_iter)

每一步训练时，在 [train.py (line 137)]

observation, actions = batch
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
loss = jnp.mean(chunked_loss)

核心训练代码在 Pi0.compute_loss 里，位置是 [pi0.py (line 189)]

它做的核心事情是 flow matching：
真实动作 actions: [16, 50, 32]
随机噪声 noise:   [16, 50, 32]
随机时间 t:       [16]

构造带噪动作:
  x_t = t * noise + (1 - t) * actions

训练目标速度:
  u_t = noise - actions

然后模型要预测 v_t，让它接近 u_t。


训练 loss 就是在 normalized 的 32 维动作空间里算的。训练过程中不会使用：Unnormalize 和 R1LiteOutputs
→ R1LiteRepack
→ R1LiteInputs
→ Normalize
→ InjectDefaultPrompt
→ ResizeImages
→ TokenizePrompt
→ PadStatesAndActions
→ model.compute_loss


7） 真正推理时不是 compute_loss；而是 而是 [sample_actions (line 217)]./src/openpi/models/pi0.py

推理流程大概是：

observation
  → preprocess_observation(train=False)
  → embed_prefix，只算一次图像和 prompt prefix，并缓存 KV cache
  → 从随机 noise 初始化 actions: [B, 50, 32]
  → 循环 num_steps 次
      当前 x_t + time
      → embed_suffix
      → LLM 预测 v_t
      → x_t = x_t + dt * v_t
  → 得到 x_0，也就是模型预测动作 [B, 50, 32]

并且需要进行反归一化 并 裁剪到14维度



#### 四，模型推理

####  1）先启动sever

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=r1lite_pack_phone_abs_joint_crop_head_image_0707 \
  --policy.dir=checkpoints/r1lite_pack_phone_abs_joint_crop_head_image_0707/r1lite_pack_phone_new_state1/14999


主要功能是
- 加载训练模型 config + checkpoint：找到TrainConfig中的模型配置，再加载 checkpoint 模型参数
- norm_stats + policy transforms：尝试从 checkpoint 目录下加载 norm stats，根据data_config 创建 transforms
- 客户端请求时：接收一帧 observation → transform 成模型输入 → sample_actions 生成动作 → 反归一化/裁剪输出 → 返回客户端


整个流程是：

1）工厂函数 
位于/src/openpi/policies/policy_config.py
主要功能是：加载 checkpoint，组装 Policy 输入输出格式
→ create_trained_policy(...)
    → 加载 checkpoint
    → return Policy(model, transforms, output_transforms)

→ 读取 checkpoint_dir
→ 判断 JAX/PyTorch checkpoint
→ 加载 model 权重
→ 创建 data_config
→ 加载 norm_stats
→ 拼 input transforms
→ 拼 output transforms
→ return Policy(...)



2）客户端请求
→ websocket server 默认监听 0.0.0.0:8000；

3）执行推理：每来一帧 observation 执行一次：做 transform，调用 model.
→ policy.infer(obs)
    → self._input_transform(obs)
    → 加 batch 维
    → Observation.from_dict(inputs)
    → self._sample_actions(...) 调用pi0.py中的model.sample_actions 生成动作
    → 去 batch 维
    → self._output_transform(outputs) 反归一化  并且actions [50, 32] → actions [50, 14]
    → return outputs

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


##### obs输入的数据格式是什么样的？

针对训练时的数据输入格式：
训练数据 transform 在 [data_loader.py (line 173)](/home/robot/wangbo/project/VLA_own/openpi/src/openpi/training/data_loader.py:173) 的 transform_dataset()：

return TransformedDataset(
    dataset,
    [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ],
)

所以训练顺序是：
→ data_config.repack_transforms
→ data_config.data_transforms
→ Normalize
→ data_config.model_transform
repack_transforms = Group(inputs=[r1lite_policy.R1LiteRepack()])

所以训练实际是：
→ R1LiteRepack
→ R1LiteInputs
→ Normalize
→ InjectDefaultPrompt
→ ResizeImages
→ TokenizePrompt
→ PadStatesAndActions

因此训练时可以吃 LeRobot 风格的字段，比如：
observation.images.cam_high
observations.state.qpos
action.qpos
task_index / prompt

针对sever的数据输入配置：

首先整个流程涉及两个函数：

create_trained_policy()
  只执行一次：加载 checkpoint，组装 Policy
→ 读取 checkpoint_dir
→ 判断 JAX/PyTorch checkpoint
→ 加载 model 权重
→ 创建 data_config
→ 加载 norm_stats
→ 拼 input transforms
→ 拼 output transforms
→ return Policy(...)



create_trained_policy() 是“把已经训练好的 checkpoint 包成一个可推理的 Policy”的函数，包括输入输出；

def create_trained_policy(train_config, checkpoint_dir, ...):
    ...
    model = train_config.model.load(...)
    ...
    return _policy.Policy(
        model,
        transforms=[...],
        output_transforms=[...],
    )

由于 serve_policy.py 没额外传 repack_transforms，这里就是空 Group()。所以 server 实际输入链是：

-》空 repack_transforms
→ InjectDefaultPrompt
→ R1LiteInputs
→ Normalize
→ InjectDefaultPrompt
→ ResizeImages
→ TokenizePrompt
→ PadStatesAndActions

针对这块的特定的输入格式是适配R1LiteInputs格式；

server 输入必须已经整理成 R1LiteInputs 能直接处理的格式：

    {
        "images": {
            "head": ...,
            "left_wrist": ...,
            "right_wrist": ...,
        },
        "state": np.ndarray,   # [14]
        "prompt": "...",
    }


- 客户端发一个 observation，格式是 msgpack_numpy。
- 当前server 调用 policy.infer(obs)。
- server 返回 action dict，里面一般有 
actions 其中actions shape ~= [action_horizon, action_dim] action_horizon=10，R1Lite joint 输出最后会裁成 14 维，所以典型输出是：[10, 14]
policy_timing 模型核心推理时间
server_timing 整个policy.infer 整体时间，包括包括输入的transform、模型推理、 输出 transform 反归一化


### 2）启动 Rollout  启动

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