from PIL import Image
import pandas as pd
from pathlib import Path
import os
from easyfsl.datasets.default_configs import default_transform
from easyfsl.datasets import FewShotDataset

class BreakHis(FewShotDataset):
    def __init__(
        self,
        root,
        magnification = 40,
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
        self.test   = ['adenosis', 'tubular_adenoma', 'phyllodes_tumor']
        self.split  = split
        self.mag    = magnification
        self.data   = self.load_specs(root, split, magnification)

        self.transforms = transforms or default_transform(image_size, training=training)

    @staticmethod
    def load_specs(root, split, mag):
        data = pd.read_csv(os.path.join(root, 'gnd.csv'), index_col=0)
        
        data = data[data.mag == mag]
        class_names = list(data.label.unique())

        if split == 'train':
            data = data[~data.label.isin(['adenosis', 'tubular_adenoma', 'phyllodes_tumor'])]
        else:
            data = data[data.label.isin(['adenosis', 'tubular_adenoma', 'phyllodes_tumor'])]
            
        label_mapping = {name: class_names.index(name) for name in class_names}
        
        data.label.replace(label_mapping, inplace=True)
        data.reset_index(inplace=True)

        return data

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, '/'.join(self.data.iloc[index].path.split('/')[1:]))).convert("RGB")
        img = self.transforms(img)

        label = self.data.iloc[index].label

        return img, label

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return list(self.data.label)