import numpy as np
import torch
from utils.loggers import CompositeLogger, NeptuneLogger, PrintLogger
from utils.utils_args import parse_args_uncond
from metrics import evaluate_model_uncond
import logging
from data_provider.data_provider import data_provider
from utils.utils import create_model_name_and_dir, log_config_and_tags
from utils.utils_vis import prepare_data, PCA_plot, TSNE_plot, density_plot, jensen_shannon_divergence
from data_provider.combined_datasets import dataset_list
from itertools import islice
from importlib import import_module

import matplotlib.pyplot as plt


def main(args):
    # Set up basic attributes
    args.finetune = not args.pretrain
    args.trained_on_datasets = [dataset for dataset in dataset_list if dataset in map(lambda s: s['name'], args.datasets)]
    
    # Model name and directory
    name = create_model_name_and_dir(args, "finetune" if args.finetune else "")
    with CompositeLogger([NeptuneLogger(), PrintLogger()]) if args.neptune \
            else CompositeLogger([PrintLogger()]) as logger:
        
        args.tags.append('visualization')
        # log config and tags
        log_config_and_tags(args, logger, name, len(args.train_on_datasets) > 1)

        # Setup Data
        dataset_loader, samplers, trainsets, metadatas = data_provider(args)
        args.n_classes = dataset_loader.num_datasets
        
        # Setup model
        assert not(args.model_ckpt == None), "Must set Model checkpoint"
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        handler = import_module(args.handler).Handler(args=args, rank=args.device)

        scores_mean = {'disc_mean': [], 'disc_std': [], 'pred_mean': [], 'pred_std': [], 'context_fid': []}

        # Evaluate + visualize
        for dataset in args.train_on_datasets:
            args.dataset = dataset
            testset, class_label = dataset_loader.gen_dataloader(dataset)
            handler.model.eval()
            with torch.no_grad():
                generated_set = handler.sample(len(testset), class_label, metadatas[dataset])
            generated_set = generated_set.cpu().detach().numpy()
            real_set = testset.cpu().detach().numpy()
            scores = evaluate_model_uncond(real_set, generated_set, dataset, args.device, args.eval_metrics, base_path=args.ts2vec_dir)
            scores_mean['disc_mean'].append(scores[f'disc_mean'])
            scores_mean['disc_std'].append(scores[f'disc_std'])
            scores_mean['pred_mean'].append(scores[f'pred_mean'])
            scores_mean['pred_std'].append(scores[f'pred_std'])
            scores_mean['context_fid'].append(scores[f'context_fid'])
            for key, value in scores.items():
                logger.log(f'test/{dataset}_{key}', value)
            logging.info("Data generation is complete")
            prep_ori, prep_gen, sample_num = prepare_data(real_set, generated_set)

            # PCA Analysis
            PCA_plot(prep_ori, prep_gen, sample_num, logger, args)
            # Do t-SNE Analysis together
            TSNE_plot(prep_ori, prep_gen, sample_num, logger, args)
            # Density plot
            density_plot(prep_ori, prep_gen, logger, args)
            # jensen shannon divergence
            jensen_shannon_divergence(prep_ori, prep_gen, logger)
            # Plot some sampled data
            for i, ts in islice(enumerate(np.transpose(generated_set, axes=(0,2,1))), 4):
                for n, channel in enumerate(ts):
                    fig = plt.figure()
                    plt.plot(channel)
                    logger.log(f"gen_channel_{dataset}_{n}", fig)
            
            for i, ts in islice(enumerate(np.transpose(generated_set, axes=(0,2,1))), 4):
                for n, channel in enumerate(ts):
                    fig = plt.figure()
                    plt.plot(channel)
                    logger.log(f"channel_{dataset}_{n}", fig)
        logger.log(f'test/disc_mean', np.mean(scores_mean['disc_mean']))
        logger.log(f'test/disc_std', np.mean(scores_mean['disc_std']))
        logger.log(f'test/pred_mean', np.mean(scores_mean['pred_mean']))
        logger.log(f'test/pred_std', np.mean(scores_mean['pred_std']))
        logger.log(f'test/context_fid', np.mean(scores_mean['context_fid']))

if __name__ == '__main__':
    args = parse_args_uncond()  # load unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)