import argparse
from moses.organ.metrics_reward import MetricsReward


def get_parser(parser=None):
    def restricted_float(arg):
        if float(arg) < 0 or float(arg) > 1:
            raise argparse.ArgumentTypeError(
                '{} not in range [0, 1]'.format(arg)
            )
        return float(arg)

    def conv_pair(arg):
        if arg[0] != '(' or arg[-1] != ')':
            raise argparse.ArgumentTypeError('Wrong pair: {}'.format(arg))

        feats, kernel_size = arg[1:-1].split(',')
        feats, kernel_size = int(feats), int(kernel_size)

        return feats, kernel_size

    if parser is None:
        parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in generator '
                                'and discriminator')
    model_arg.add_argument('--hidden_size', type=int, default=512,
                           help='Size of hidden state for lstm '
                                'layers in generator')
    model_arg.add_argument('--num_layers', type=int, default=2,
                           help='Number of lstm layers in generator')
    model_arg.add_argument('--dropout', type=float, default=0,
                           help='Dropout probability for lstm '
                                'layers in generator')
    model_arg.add_argument('--discriminator_layers', nargs='+', type=conv_pair,
                           default=[(100, 1), (200, 2), (200, 3),
                                    (200, 4), (200, 5), (100, 6),
                                    (100, 7), (100, 8), (100, 9),
                                    (100, 10), (160, 15), (160, 20)],
                           help='Numbers of features for convalution '
                                'layers in discriminator')
    model_arg.add_argument('--discriminator_dropout', type=float, default=0,
                           help='Dropout probability for discriminator')
    model_arg.add_argument('--reward_weight', type=restricted_float,
                           default=0.7,
                           help='Reward weight for policy gradient training')

    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--generator_pretrain_epochs', type=int,
                           default=50,
                           help='Number of epochs for generator pretraining')
    train_arg.add_argument('--discriminator_pretrain_epochs', type=int,
                           default=50,
                           help='Number of epochs for '
                                'discriminator pretraining')
    train_arg.add_argument('--pg_iters', type=int, default=1000,
                           help='Number of inerations for policy '
                                'gradient training')
    train_arg.add_argument('--n_batch', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-4,
                           help='Learning rate')
    train_arg.add_argument('--n_jobs', type=int, default=8,
                           help='Number of threads')

    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')
    train_arg.add_argument('--max_length', type=int, default=100,
                           help='Maximum length for sequence')
    train_arg.add_argument('--clip_grad', type=float, default=5,
                           help='Clip PG generator gradients to this value')
    train_arg.add_argument('--rollouts', type=int, default=16,
                           help='Number of rollouts')
    train_arg.add_argument('--generator_updates', type=int, default=1,
                           help='Number of updates of generator per iteration')
    train_arg.add_argument('--discriminator_updates', type=int, default=1,
                           help='Number of updates of discriminator '
                                'per iteration')
    train_arg.add_argument('--discriminator_epochs', type=int, default=10,
                           help='Number of epochs of discriminator '
                                'per iteration')
    train_arg.add_argument('--pg_smooth_const', type=float, default=0.1,
                           help='Smoothing factor for Policy Gradient logs')

    parser.add_argument('--n_ref_subsample', type=int, default=500,
                        help='Number of reference molecules '
                             '(sampling from training data)')
    parser.add_argument('--additional_rewards', nargs='+', type=str,
                        choices=MetricsReward.supported_metrics, default=[],
                        help='Adding of addition rewards')
    
    # Docking
    parser.add_argument('--target', type=str, default='usp7',
                        choices=['fa7', 'parp1', '5ht1b', 'usp7', 'abl1', 'fkb1a'])
    parser.add_argument('--receptor_path', type=str, required=True)
    parser.add_argument('--vina_path', type=str, required=True)
    parser.add_argument('--temp_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--num_modes', type=int, default=10)
    parser.add_argument('--num_sub_proc', type=int, default=1)
    parser.add_argument('--n_conf', type=int, default=3)

    return parser


def get_docking_config(args):
    if args.target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
    elif args.target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
    elif args.target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)
    elif args.target == 'usp7':
        if args.pocket_id == 0:
            box_center = (2.860, 4.819, 92.848)
            box_size = (17.112, 17.038, 14.958)
        elif args.pocket_id == 1:
            box_center = (27.413, 1.55, 29.902)
            box_size = (16.221, 16.995, 17.858)
    elif args.target == 'abl1':
        box_center = (16.496, 14.747, 3.999)
        box_size = (14.963, 8.151, 5.892)
    elif args.target == 'fkb1a':
        box_center = (-35.137, 39.04, 32.495)
        box_size = (8.453, 13.483, 8.112)
    else:
        raise ValueError(f'Unknown receptor {args.target}.')

    docking_config = {
        'receptor_file': args.receptor_path,
        'box_center': box_center,
        'box_size': box_size,
        'vina_program': args.vina_path,
        'temp_dir': args.temp_dir,
        'exhaustiveness': args.exhaustiveness,
        'num_sub_proc': args.num_sub_proc,
        'num_modes': args.num_modes,
        'timeout_gen3d': None,
        'timeout_dock': None,
        'seed': args.seed,
        'n_conf': args.n_conf,
        'error_val': 99.9,
        'alpha': args.alpha
    }

    return docking_config


def get_config():
    parser = get_parser()
    args = parser.parse_known_args()[0]
    args.metrics_configs = {'docking': get_docking_config(args)}
    return args
