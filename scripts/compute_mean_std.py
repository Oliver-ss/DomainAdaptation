from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())
from common import config
from data import make_data_loader
import data.spacenet as spacenet
from torch.utils.data import DataLoader
import json
import torch

def compute_mean_variance(city='Vegas'):
    #ini_dist = {'mean': (0.,0.,0.), 'std': (1., 1., 1.)}
    train_set = spacenet.Spacenet(city=city, split='train', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.train_num_workers, drop_last=True)
    mean = 0.
    std = 0.
    nb_samples = 0.
    all_data = None
    for sample in tqdm(loader):
        data = sample['image']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        #if all_data is None:
        #    all_data = data
        #else:
        #    torch.cat((all_data, data), 0)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    #mean = data.mean(2).sum(0)
    #std = data.std(2).sum(0)
    mean = mean.numpy().astype(float)
    std = std.numpy().astype(float)
    print('city: ', city)
    print('mean: ', mean)
    print('std:', std)
    return (mean[0],mean[1],mean[2]), (std[0], std[1], std[2])

if __name__ == '__main__':
    cities = ['Vegas', 'Shanghai', 'Paris', 'Khartoum']
    output = {}
    for city in cities:
        mean, std = compute_mean_variance(city)
        output[city] = {'mean':mean, 'std': std}
    with open('../../scripts/mean_std.json', 'w') as f:
        json.dump(output, f)
