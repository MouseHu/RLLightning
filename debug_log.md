*created by lzy, will be deleted after finished*
command:
python train.py --env_type atari --load_json configs/a2c.json --env_name Pong-ram-v0 --total_steps 100
python train.py --env_type atari --load_json configs/ddpg.json --env_name Pendulum-v0 --total_steps 100

4.20

update:
1. add support for atari ram env
2. add a runable a2c agent

为什么有的epoch会跑得特别快？
一个train_step是优化一个batch

a2c:on-policy batch_size=n_steps*n_env

适配a2c的努力：
1. 由于开始输出及其不稳定，对环境的input进行了归一化，并且没有使用layernorm(感觉会改变数据分布)
2. 由于优化始终卡在某个瓶颈，调小了actor的学习率，增加了网络的复杂度
3. 参考stable_baseline的实现，将每个epoch的batch_size尽量降低

4.21

现象：各自有一些收敛的evidence，但是分别在20000（更改网络结构后）和30000（更改网络结构前）epoch左右出现了entropy loss的下降，看起来需要将entropy coef增大到0.01。

同时开始测试PPO.
debuging DDPG....

4.22
代码跑太慢了......
降低ppo entropy coef到0.003后很快收敛到局部极值了，看来不能降。

成果：
1. 写了能跑的PPO和A2C
2. 添加了对vector env的支持并且做了obs的归一化
3. 正在debugging DDPG
4. 研究了一下tianshou的并行环境实现，感觉可以抄过来

问题：
1. 关于模块设计与耦合
- 当前模块之间耦合过于强，比如agent.step中，存在state不匹配的风险，而且buffer似乎应该交给leaner管理

- network-agent-leaner要全部一一耦合吗？
    - network似乎有重复？比如DDPG的network全是linear的

- 参考:stable-baseline的runner和agent-leaner一一对应
- tianshou的collector和policy（agent-leaner）是分开的

讨论结果：
1. 找PPO经典环境（TD3）

2. agent可以负责generate_data.

4. network可以整理一下，leaner可以分为off-policy AC,Q,PG

impala，R2D2, A3C 研究一下Agent和采数据到底是什么关系。

训练正常速度：半天5million

下一步：在atari上调通所有基本算法；整理网络；整理leaner；支持并行环境；支持分布式训练；支持调超参。