from utils.utils import get_config, get_evaluator, get_guidance, get_network
from pipeline import BasePipeline
import torch
import logger
from copy import deepcopy

if __name__ == '__main__':
    # Please tsee utils/config.py for the complete argument lists
    args = get_config()
    ## prepare core modules based on configs ##
    
    # Unconditional generative model
    network = get_network(args)
    # guidance method encoded by prediction model
    guider = get_guidance(args, network)
    
    bon_guider = None
    if hasattr(args, 'bon_guidance') and args.bon_guidance:
        bon_args = deepcopy(args)
        bon_args.guide_networks = args.bon_guidance.split('+')
        bon_guider = get_guidance(bon_args, network)
    
    # evaluator for generated samples
    try:
        evaluator = get_evaluator(args)
    except NotImplementedError:
        evaluator = None

    pipeline = BasePipeline(args, network, guider, evaluator, bon_guider=bon_guider)

    samples = pipeline.sample(args.num_samples)
    logger.log_samples(samples)
    
    # release torch occupied gpu memory
    torch.cuda.empty_cache()
    
    # FID 계산 비활성화 (샘플링만 수행)
    # metrics = evaluator.evaluate(samples)
    # if metrics is not None: # avoid rewriting metrics to json
    #     logger.log_metrics(metrics, save_json=True)