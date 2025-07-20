from metrics.discriminative_torch import discriminative_score_metrics
from .context_fid import Context_FID
import numpy as np
import torch
from tqdm import tqdm

def evaluate_model_uncond(real_sig, gen_sig, dataset, device, eval_metrics=['disc'], metric_iteration=10, base_path=None):

    if 'disc' in eval_metrics:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        disc_res = list()
        for _ in tqdm(range(metric_iteration), desc='Discriminative score evaluation'):
            dsc = discriminative_score_metrics(real_sig, gen_sig, device)
            disc_res.append(dsc)
        disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)
    else:
        disc_mean, disc_std = -1, -1

    if 'contextFID' in eval_metrics:
        context_fid = Context_FID(real_sig, gen_sig, dataset, device, base_path)
    else:
        context_fid = -1

    if 'pred' in eval_metrics:
        from metrics.predictive_metrics import predictive_score_metrics
        predictive_score = list()
        for _ in tqdm(range(metric_iteration), desc="Predictive score evaluation"):
            temp_pred = predictive_score_metrics(real_sig, gen_sig)
            predictive_score.append(temp_pred)
        pred_mean, pred_std = np.round(np.mean(predictive_score), 4), np.round(np.std(predictive_score), 4)
    else:
        pred_mean, pred_std= -1, -1

    return {
        f'disc_mean':disc_mean,
        f'disc_std':disc_std,
        f'pred_mean':pred_mean,
        f'pred_std':pred_std,
        f'context_fid':context_fid}