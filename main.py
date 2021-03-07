import pytorch_lightning as pl

from utils.common import get_args, setup, get_path

args = get_args()
env, eval_env, replay_buffer, agent, learner = setup(args)

trainer = pl.Trainer(gpus=1, accelerator='dp', max_steps=args.total_steps,
                     default_root_dir=get_path(args), check_val_every_n_epoch=2500, log_every_n_steps=1)

trainer.early_stop_callback = False  # disable early stopping

trainer.fit(learner)
