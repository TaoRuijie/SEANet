import os, numpy, torch, warnings, glob, random

EPS = 1e-6
def init_system(args):
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok = True)

    args.modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
    args.modelfiles.sort()
    args.score_file = open(args.score_save_path, "a+")
    torch.set_printoptions(precision=4,sci_mode=False)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return args

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr