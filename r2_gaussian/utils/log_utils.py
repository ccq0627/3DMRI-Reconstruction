import os
import sys
import uuid
import os.path as osp
from argparse import Namespace
import yaml, json
import os.path as osp
from datetime import datetime
from r2_gaussian.arguments import OptimizationParams,ModelParams
from tqdm import tqdm
import logging

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.append("./")
from r2_gaussian.utils.cfg_utils import args2string


def prepare_output_and_logger(args):
    # Update model path if not specified
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = osp.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(osp.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save to yaml
    args_dict = vars(args)
    with open(osp.join(args.model_path, "cfg_args.yml"), "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Create Tensorboard writer
    tb_writer = None
    # cancel tensorboard
    # if TENSORBOARD_FOUND:
    if False:
        tb_writer = SummaryWriter(args.model_path)
        tb_writer.add_text("args", args2string(args_dict), global_step=0)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def setup_experiment_folder(args, base_dir="MRIdata/outputs"):
    time_str = datetime.now().strftime("%m-%d_%H-%M")
    exp_name = f"exp_{time_str}_iter{args.iterations}_{args.accelerate_factor}_{'wi' if args.use_image_loss else 'woi'}"

    exp_dir = osp.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    config_path = osp.join(exp_dir, "exp_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
        
    return exp_dir


def prepare_tqdm_write_logger(log_path="log.txt", level=logging.INFO):
    """Patch tqdm.write so messages are also appended to a log file."""
    logger = logging.getLogger("tqdm_write_logger")
    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(osp.dirname(log_path) or ".", exist_ok=True)

    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if getattr(tqdm, "_write_logger_installed", False):
        return

    original_write = tqdm.write

    def write_and_log(cls, s, file=None, end="\n", nolock=False):
        original_write(s, file=file, end=end, nolock=nolock)
        logger.log(level, str(s).rstrip("\n"))

    tqdm.write = classmethod(write_and_log)
    tqdm._write_logger_installed = True


