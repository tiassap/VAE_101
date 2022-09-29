import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, models,datasets
import logging

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def join(loader, node):
    baseline_map = {'True':'baseline', 'False':'no-baseline'}
    seq = loader.construct_sequence(node)
    seq = [baseline_map[str(x)] if (str(x) in baseline_map.keys()) else x for x in seq]
    return ''.join([str(i) for i in seq])

def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def get_data_loaders(data_root, batch_size,num_workers):
    '''
        returns data loader. 
    '''
    train_test_transforms = transforms.Compose([
        # The following re-scales the image tensor values to be between 0-1: image_tensor /= 255
        transforms.ToTensor()
    ])
    
    dataset_train = datasets.MNIST(root=data_root, train=True, download=True, transform=train_test_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader