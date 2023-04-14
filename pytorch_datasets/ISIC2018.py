from PIL import Image
import pandas as pd
from pathlib import Path
import os
from easyfsl.datasets.default_configs import default_transform
from easyfsl.datasets import FewShotDataset

class ISIC2018(FewShotDataset):
    def __init__(
        self,
        root,
        image_size = 224,
        split = 'train',
        transforms = None,
        training = True
        ):
        '''
        Args:
            root: directory where all the images are
            image_size: images returned by the dataset will be square images of the given size
            split: train|test
            transform: torchvision transforms to be applied to images. If none is provided,
                we use some standard transformations including ImageNet normalization.
                These default transformations depend on the "training" argument.
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip. Only used if transforms = None.
        '''
        self.root   = Path(root)
        self.test   = ('AKIEC', 'VASC', 'DF')
        self.split  = split
        self.data   = self.load_specs(root, split)

        self.transforms = transforms or default_transform(image_size, training=training)

    @staticmethod
    def load_specs(root, split):
        data = pd.read_csv(os.path.join(root, 'gnd', 'train_gnd.csv'), index_col=0)
        
        data = data.idxmax(axis=1)
        class_names   = list(data.unique())

        if split == 'train':
            data = data[data.isin(['MEL', 'NV', 'BCC', 'BKL'])]
        else:
            data = data[data.isin(['AKIEC', 'VASC', 'DF'])]
            
        label_mapping = {name: class_names.index(name) for name in class_names}

        return data.replace(label_mapping)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'train', f'{self.data.index[index]}.jpg')).convert("RGB")
        img = self.transforms(img)

        label = self.data[index]

        return img, label

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return list(self.data)