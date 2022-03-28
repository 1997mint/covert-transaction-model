import numpy as np
from SeqGAN.train import Trainer
from SeqGAN.get_config import get_config

config = get_config('config.ini')

trainer = Trainer(config["batch_size"],
                config["max_length"],
                config["g_e"],
                config["g_h"],
                config["d_e"],
                config["d_h"],
                config["d_dropout"],
                path_pos=config["path_pos"],
                path_neg=config["path_neg"],
                g_lr=config["g_lr"],
                d_lr=config["d_lr"],
                n_sample=config["n_sample"],
                generate_samples=config["generate_samples"])




trainer.load_cnn(config["cnn_weights_path"])
trainer.load(config["g_weights_path"], config["d_weights_path"])
trainer.generate_txt(config["g_test_path"], config["generate_samples"])

