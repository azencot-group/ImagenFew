import scipy
import numpy as np
import torch
import os

from metrics.models.ts2vec.ts2vec import TS2Vec

base_path = "/cs/cs_groups/azencot_group/diffusion_foundation/TS2VEC_ckpts"

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def Context_FID(ori_data, generated_data, dataset, device, base_path=None):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=device, batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    
    if base_path:
        model_path = os.path.join(base_path, f"{dataset}_{'_'.join(str(dim) for dim in ori_data.shape)}.ckpt")
        rep_path = os.path.join(base_path, f"{dataset}_{'_'.join(str(dim) for dim in ori_data.shape)}_rep.ckpt")
        if os.path.exists(model_path) and os.path.exists(rep_path):
            model.load(model_path)
            ori_represenation = torch.load(rep_path)
        else:
            model.fit(ori_data, verbose=False)
            model.save(model_path)
            ori_represenation = model.encode(ori_data, encoding_window='full_series')
            torch.save(ori_represenation, rep_path)
    else:
        model.fit(ori_data, verbose=False)
        model.save(model_path)
        ori_represenation = model.encode(ori_data, encoding_window='full_series')
    
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results

def load_model(path, model):
    model.net.load_state_dict(torch.load(path))
    return model

def create_and_save_model(path, model):
    torch.save(model.net.state_dict(), path)
    return model