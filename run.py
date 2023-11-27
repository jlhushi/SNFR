import argparse
import itertools
import time
import torch

from model import SNFR
from utils_demo.get_mask import get_mask
from utils_demo.util import cal_std
from utils_demo.logger_ import get_logger
from utils_demo.datasets import *
from configure.configure_clustering import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

datasets = ["Caltech101-20", "Scene_15", "NoisyMNIST", "LandUse_21", "washington", "CCV"]
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')
args = parser.parse_args()
selected_dataset = datasets[args.dataset]

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    logger = get_logger()  # 创建一个新的 logger 对象

    config = configure_and_log(get_default_config(), dataset, args)

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % k)
            for (g, z) in v.items():
                logger.info("\t%s = %s" % (g, z))
            logger.info("%s: %s" % (k, v))

    # Load and initialize data
    x1_train_raw, x2_train_raw, Y_list = load_and_initialize(config)

    fold_acc, fold_nmi, fold_ari = [], [], []
    start_time = time.time()
    accumulated_metrics = collections.defaultdict(list)
    optimizer = torch.optim.Adam(SNFR_model.parameters(), lr=config['training']['lr'])

    for data_seed in range(1, args.test_time + 1):
        start = time.time()

        np.random.seed(data_seed)
        seed = data_seed if config['missing_rate'] == 0 else config['seed']

        mask = get_mask(2, x1_train_raw.shape[0], cdonfig['missing_rate'])
        x1_train = torch.from_numpy(x1_train_raw * mask[:, 0][:, np.newaxis]).float().to(device)
        x2_train = torch.from_numpy(x2_train_raw * mask[:, 1][:, np.newaxis]).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Training
        acc, nmi, ari = SNFR_model.train_clustering(config, logger, accumulated_metrics, x1_train, x2_train, Y_list,
                                                    mask,
                                                    optimizer, device)
        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        print(time.time() - start)

    logger.info('--------------------Training over--------------------')
    acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time} 秒")

def configure_and_log(config, dataset, args):
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    return config

def load_and_initialize(config):
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    return x1_train_raw, x2_train_raw, Y_list

if __name__ == '__main__':
    main()
