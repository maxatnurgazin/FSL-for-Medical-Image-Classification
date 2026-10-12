from typing import Any, Callable, Optional
from easyfsl.datasets.default_configs import default_transform
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

class Papsmear(ImageFolder):
    def __init__(
        self, 
        root: str, 
        training: bool, 
        image_size: int = 224, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
        loader: Callable[[str], Any] = default_loader,
    ):
        transforms = transform or default_transform(image_size, training=training)
        super().__init__(root, transforms,  loader = loader, is_valid_file = is_valid_file, target_transform = None,)
    def get_labels(self):
        return self.targets
    
def is_valid_file(path : str):
    return path.split('.')[0][-1] != 'd'