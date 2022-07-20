from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # tensorboard visualization parameters
        parser.add_argument('--display_freq', type=int, default=320, help='frequency of showing training results on screen')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--print_freq', type=int, default=160, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=300, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs_warmup', type=int, default=5, help='number of epochs to warm up learning rate, only for warmup policy')
        parser.add_argument('--n_epochs', type=int, default=15, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=140, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate for basenet.')
        parser.add_argument('--lr_policy', type=str, default='warmup', help='learning rate policy. [linear | step | plateau | cosine | warmup]')
        parser.add_argument('--lr_policy_after', type=str, default='linear', help='learning rate for after policy used in warmup policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
