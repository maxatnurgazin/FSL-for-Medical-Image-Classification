# %%
import random
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
import torchvision.transforms as transforms
import re
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from pytorch_datasets import ISIC2018, BreakHis, Papsmear
from pytorch_datasets import augmentations

from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate
from torch.utils.data import DataLoader
from easyfsl.datasets import MiniImageNet
from easyfsl.datasets.default_configs import default_transform
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier, MatchingNetworks
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math

DEVICE = "cuda"
IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

def protonet(models, dataset, algo, mag=100,):
    print(mag, algo)
    # %%
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # %%
    cutout_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        augmentations.Cutout(n_holes=1, length=64),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZATION),
    ])
    train_transform = default_transform(image_size=224, training=True)
    test_transform  = default_transform(image_size=224, training=False)

    # %%
    # Training Parameters
    n_way   = 2
    n_shot  = 3
    n_query = 10

    n_episodes          = 40000
    n_tasks_per_epoch   = 500
    n_validation_tasks  = 200

    n_workers = 6

    # Selecting augmentation
    augments = [None, 'cutout', 'mixup', 'cutmix']
    augmentation  = augments[0]

    if augmentation == 'cutout':
        transform = cutout_transform
    else:
        transform = train_transform


    # Selecting a dataset
    datasets  = {0: 'ISIC2018', 1: 'BreakHis', 2: 'Papsmear'}
    dataset   = datasets[dataset] # Dataset ID

    if dataset == 'ISIC2018':
        root  = 'C:/Datasets/ISIC2018'
        train_set = ISIC2018(root=root, split='train', training=True, transforms=transform)
        val_set   = ISIC2018(root=root, split='validation', training=False)
    elif dataset == 'BreakHis':
        root          = 'C:/Datasets/BreaKHis_v1'
        magnification = mag

        train_set = BreakHis(root=root, magnification=magnification, split='train', training = True,transforms=transform)
        val_set   = BreakHis(root=root, magnification=magnification, split='validation', training=False)
    elif dataset == 'Papsmear':
        train_root = r"C:\Datasets\smear2005\New database pictures\train"
        test_root  = r"C:\Datasets\smear2005\New database pictures\test"

        train_set = Papsmear(root=train_root, training=True, transform=transform)
        val_set   = Papsmear(root=test_root, training=False)


    train_sampler = TaskSampler(
        train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
    )
    val_sampler = TaskSampler(
        val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
    )

    # Finally, the DataLoader. We customize the collate_fn so that batches are delivered
    # in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    for model_name in models:
        train(model_name, algo, train_loader, val_loader, val_set, augmentation)
models = ['mobilevitv2_050', 
          'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
          'swin_base_patch4_window7_224',
          'resnet50']
def train(model_name, algo, train_loader, val_loader, val_set, augmentation):
    vit = timm.create_model(model_name, pretrained=True, in_chans=3)
    if re.match('^vit_', model_name) or model_name == 'deit_base_patch16_224':
        vit.head = nn.Identity() # vit
    elif re.match('^swin_', model_name):
        vit.head.fc = nn.Identity() # swin
    elif re.match('^mobilevitv2', model_name):
        vit.fc = nn.Identity() # mvit mobilevitv2
    elif model_name == 'resnet50':
        vit.fc = nn.Identity() # resnet50
    elif model_name == 'vgg16':
        vit.head.fc = nn.Identity() # vgg16
    vit = vit.cuda()
    if model_name == 'vit_tiny_patch16_224':
        fd = 192
    elif model_name in ['mobilevitv2_050', 'vit_small_patch16_224']:
        fd = 384
    elif model_name in ['vit_base_patch16_224', 'deit_base_patch16_224']:
        fd = 768
    elif model_name == 'swin_base_patch4_window7_224':
        fd = 1024
    elif model_name == 'resnet50':
        fd = 2048
    elif model_name== 'vgg16':
        fd = 4096

    if algo == 'protonet':
        few_shot_classifier = PrototypicalNetworks(vit).to(DEVICE)
        LOSS_FUNCTION = nn.CrossEntropyLoss()

    elif algo == 'matching':
        few_shot_classifier = MatchingNetworks(vit, feature_dimension = fd).to(DEVICE)
        LOSS_FUNCTION = nn.NLLLoss()

    # if augmentation == 'mixup':
    #     LOSS_FUNCTION = augmentations.mixup_criterion()
    # else:

    n_epochs = 7
    scheduler_milestones = [10, 20]
    warmup_epochs = 3
    scheduler_gamma = 0.1
    learning_rate = 1e-5
    tb_logs_dir = Path(".")

    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9
    )

    warmup_scheduler = LinearLR(
        train_optimizer,
        start_factor=0.02,
        end_factor=1, 
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        train_optimizer,
        T_max=n_epochs-warmup_epochs,
        eta_min=1e-6
    )

    seq_scheduler = SequentialLR(
        train_optimizer, 
        [warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    # train_scheduler = seq_scheduler

    # %%
    def training_epoch(
        model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer, criterion, augmentation: str, alpha=0.4
    ):
        all_loss = []
        model.train()
        with tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                optimizer.zero_grad() 
                
                model.process_support_set(
                    support_images.to(DEVICE), support_labels.to(DEVICE)
                )
                
                classification_scores = model(query_images.to(DEVICE))
                loss = criterion(classification_scores, query_labels.to(DEVICE))

                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())

                tqdm_train.set_postfix(loss=mean(all_loss))

        return mean(all_loss)

    # %%
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def training_epoch_tensor(
        model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer, criterion, augmentation: str, alpha=0.4
    ):
        all_loss = []
        model.train()
        with tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training", leave=False,
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                optimizer.zero_grad() 
                
                model.process_support_set(
                    support_images.to(DEVICE), support_labels.to(DEVICE)
                )
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    classification_scores = model(query_images.to(DEVICE))
                    loss = criterion(classification_scores, query_labels.to(DEVICE))

                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                all_loss.append(loss.item())

                tqdm_train.set_postfix(loss=mean(all_loss))

        return mean(all_loss)

    # %% [markdown]
    # ### Train

    # # %%
    # best_state = few_shot_classifier.state_dict()
    # best_validation_accuracy = 0.0
    # for epoch in range(n_epochs):
    #     print(f"Epoch {epoch}")
    #     average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer, LOSS_FUNCTION, augmentation)
    #     validation_accuracy = evaluate(
    #         few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
    #     )

    #     if validation_accuracy > best_validation_accuracy:
    #         best_validation_accuracy = validation_accuracy
    #         best_state = few_shot_classifier.state_dict()
    #         print("Ding ding ding! We found a new best model!")

    #     # tb_writer.add_scalar("Train/loss", average_loss, epoch)
    #     # tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

    #     # Warn the scheduler that we did an epoch
    #     # so it knows when to decrease the learning rate
    #     train_scheduler.step()

    # %%
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0

    for epoch in range(n_epochs):
        # print(f"Epoch {epoch}")
        average_loss = training_epoch_tensor(few_shot_classifier, train_loader, train_optimizer, LOSS_FUNCTION, augmentation)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = few_shot_classifier.state_dict()
            # print("Ding ding ding! We found a new best model!")

        # tb_writer.add_scalar("Train/loss", average_loss, epoch)
        # tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    # %%
    latest = few_shot_classifier.state_dict()

    # %%
    few_shot_classifier.load_state_dict(best_state)

    # %%
    # few_shot_classifier = torch.load("E:\Models\protonet_checkpoints\\vit_tiny_patch16_224.pt")

    # %% [markdown]
    # ## Evaluation

    # %%
    #Evaluation Loop
    ways   = [2, 3]
    shots  = [3, 5, 10]
    n_query = 10

    n_validation_tasks  = 400

    n_workers = 6
    print(model_name)
    for way in ways:
        for shot in shots:
            test_sampler = TaskSampler(
                val_set, n_way=way, n_shot=shot, n_query=n_query, n_tasks=n_validation_tasks
            )

            # Finally, the DataLoader. We customize the collate_fn so that batches are delivered
            # in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
            test_loader = DataLoader(
                val_set,
                batch_sampler=test_sampler,
                num_workers=n_workers,
                pin_memory=True,
                collate_fn=test_sampler.episodic_collate_fn,
            )

            accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
            print(f"Average accuracy : {(100 * accuracy):.2f} %")

    # %%
    # accuracy = evaluate(few_shot_classifier, val_loader, device=DEVICE)
    # print(f"Average accuracy : {(100 * accuracy):.2f} %")

    # %%
    # torch.save(few_shot_classifier, f'E:/Models/protonet_checkpoints/{model_name}.pt')


