from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.loader_train = False
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args, train=True)
            self.loader_train = DataLoader(trainset, args.batch_size, shuffle=False)
            # trainset2 = getattr(module_train, args.data_train)(args, train=True)
            # self.loader_train2 = DataLoader(trainset2, 1, shuffle=False)
            evalset = getattr(module_train, args.data_train)(args, train=False)
            self.loader_eval = DataLoader(evalset, 1, shuffle=True)
        else:
            if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.' + args.data_test.lower())
                testset = getattr(module_test, args.data_test)(args, train=False)
            self.loader_test = DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
            )
