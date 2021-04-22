import pytorch_lightning as pl

from utils.common import get_args, setup, get_path
from pytorch_lightning.loggers import TensorBoardLogger
import logging



args = get_args()
print(args)

# configure logging at the root level of lightning
logger = TensorBoardLogger('tb_logs', name=args.agent)
env, eval_env, replay_buffer, agent, learner = setup(args)
print(agent)
print(learner)
trainer = pl.Trainer(gpus=1,accelerator='dp', max_steps=args.total_steps,
                     default_root_dir=get_path(args), check_val_every_n_epoch=2500, log_every_n_steps=300,
                     logger=logger,flush_logs_every_n_steps=1000)

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
trainer.early_stop_callback = False  # disable early stopping
print("fitting")
trainer.fit(learner)
