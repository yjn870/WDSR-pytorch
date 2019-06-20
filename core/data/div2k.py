import glob
import h5py
from tqdm import tqdm
from .utils import load_img, img2np, get_patch, np2tensor, normalize


class DIV2K(object):
    def __init__(self, args, train, n_pairs=None):
        self.args = args
        self.train = train

        if not n_pairs:
            self.n_pairs = 800 if self.train else 100
        else:
            self.n_pairs = n_pairs

        if not self._is_ready():
            self._prepare()

    def _prepare(self):
        print('Prepare DIV2K dataset...')

        for phase in ['train', 'valid'] if self.train else ['valid']:
            print('Generate {} h5 file...'.format(phase))

            dataset_path = '{}/DIV2K_{}_x{}.h5'.format(self.args.dataset_dir, phase, self.args.scale)

            h5 = h5py.File(dataset_path, 'w')

            hr_group = h5.create_group('hr')
            lr_group = h5.create_group('lr')

            hr_list = sorted(glob.glob('{}/DIV2K_{}_HR/*.png'.format(self.args.dataset_dir, phase)))
            lr_list = sorted(glob.glob('{}/DIV2K_{}_LR_bicubic/X{}/*.png'.format(self.args.dataset_dir, phase,
                                                                                 self.args.scale)))

            with tqdm(total=len(hr_list)) as t:
                t.set_description('hr')
                for i, path in enumerate(hr_list):
                    hr_group.create_dataset(str(i), data=img2np(load_img(path)))
                    t.update()

            with tqdm(total=len(lr_list)) as t:
                t.set_description('lr')
                for i, path in enumerate(lr_list):
                    lr_group.create_dataset(str(i), data=img2np(load_img(path)))
                    t.update()

            h5.close()

            print('path=', dataset_path)

        print()

    def _is_ready(self):
        try:
            for phase, num_images in [('train', 800), ('valid', 100)] if self.train else [('valid', 100)]:
                with h5py.File('{}/DIV2K_{}_x{}.h5'.format(self.args.dataset_dir, phase, self.args.scale), 'r') as h5:
                    assert len(h5['hr']) == num_images and len(h5['lr']) == num_images
        except Exception:
            return False
        return True

    def __len__(self):
        if self.train:
            repeat = self.args.batch_size * self.args.iterations_per_epoch // self.n_pairs
        else:
            repeat = 1
        return self.n_pairs * repeat

    def __getitem__(self, idx):
        idx = str(idx % self.n_pairs)

        with h5py.File('{}/DIV2K_{}_x{}.h5'.format(self.args.dataset_dir, 'train' if self.train else 'valid',
                                                   self.args.scale), 'r') as h5:
            if self.train:
                lr, hr = get_patch(h5['lr'][idx], h5['hr'][idx], self.args.patch_size, self.args.scale,
                                   self.args.augment_patch)
            else:
                lr, hr = h5['lr'][idx].value, h5['hr'][idx].value

            lr = np2tensor(lr)
            hr = np2tensor(hr)

            lr = normalize(lr, max_value=self.args.rgb_range[1])
            hr = normalize(hr, max_value=self.args.rgb_range[1])

            return lr, hr
