from mnist4molecules.config import get_train_parser as \
    get_common_train_parser, get_sample_parser as get_common_sample_parser


def get_train_parser():
    parser = get_common_train_parser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           type=bool, default=True,
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=64,
                           help='Encoder h dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.0,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=1,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0.0,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=64,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           type=bool, default=False,
                           help='If to freeze embeddings while training')

    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=64,
                           help='Batch size')
    train_arg.add_argument('--grad_clipping',
                           type=int, default=10,
                           help='Gradients clipping size')
    train_arg.add_argument('--kl_start',
                           type=int, default=1,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=0.01,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=0.01,
                           help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=1e-3,
                           help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
                           type=int, default=1,
                           help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
                           type=float, default=1e-3,
                           help='Maximum lr weight value')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')

    return parser


def get_sample_parser():
    parser = get_common_sample_parser()

    # Sample
    sample_arg = parser.add_argument_group('Sample')
    sample_arg.add_argument('--n_len',
                            type=int, default=150,
                            help='Maximum sampling len')

    return parser