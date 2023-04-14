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

from tqdm import tqdm
import timm
from pytorch_datasets import ISIC2018, BreakHis, Papsmear
from pytorch_datasets import augmentations

from easyfsl.samplers import TaskSampler
from easyfsl.methods.utils import evaluate
from torch.utils.data import DataLoader


def protonet(model_name: str, dataset: int):

    # %%
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = "cuda"
    IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    # %%
    # Hyper-Parameters
    learning_rate       = 1e-6

    n_evaluation_tasks  = 600
    n_epochs            = 15
    n_tasks_per_epoch   = 2000

    # %%
    cutout_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        augmentations.Cutout(n_holes=1, length=64),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZATION),
    ])
    simple_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

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
        transform = None


    # Selecting a dataset
    datasets  = {0: 'ISIC2018', 1: 'BreakHis', 2: 'Papsmear'}
    dataset   = datasets[dataset] # Dataset ID

    if dataset == 'ISIC2018':
        root  = 'C:/Datasets/ISIC2018'
        train_set = ISIC2018(root=root, split='train', training=True, transforms=transform)
        val_set   = ISIC2018(root=root, split='validation', training=False, transforms=transform)
    elif dataset == 'BreakHis':
        root          = 'C:/Datasets/BreaKHis_v1'
        magnification = 100

        train_set = BreakHis(root=root, magnification=magnification, split='train', training = True,transforms=transform)
        val_set   = BreakHis(root=root, magnification=magnification, split='validation', training=False, transforms=transform)
    elif dataset == 'Papsmear':
        train_root = r"C:\Datasets\smear2005\New database pictures\train"
        test_root  = r"C:\Datasets\smear2005\New database pictures\test"

        train_set = Papsmear(root=train_root, training=True, transform=transform)
        val_set   = Papsmear(root=test_root, training=False, transform=transform)


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
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    # %%
    # def show_tensor(tensor):
    #     display(transforms.ToPILImage()(tensor))

    # # %%
    # show_tensor(train_set[0][0])

    # %% [markdown]
    # ## Experiments

    # %%
    # d = iter(train_loader)

    # %%
    # support_images, support_labels, query_images, query_labels, _ = d.next()

    # %% [markdown]
    # ## Training

    # %%
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # %%
    # timm.list_models()

    # %% [markdown]
    # ### Model selection

    # %%
    from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
    from easyfsl.modules import resnet12

    # model_name = 'vit_small_patch16_224'
    # model_name = 'resnet50'
    # model_name = 'mobilevitv2_050'

    vit = timm.create_model(model_name, pretrained=True, in_chans=3)
    if re.match('^vit_', model_name):
        vit.head = nn.Identity() # vit
    elif re.match('^swin_', model_name):
        vit.head.fc = nn.Identity() # swin
    elif re.match('^mobilevitv2', model_name):
        vit.fc = nn.Identity() # mvit mobilevitv2
    elif model_name == 'resnet50':
        vit.fc = nn.Identity() # resnet50
    elif model_name == 'vgg16':
        vit.head.fc = nn.Identity() # vgg16
    # convolutional_network = resnet12()
    few_shot_classifier = PrototypicalNetworks(vit).to(DEVICE)

    # %%
    count_parameters(few_shot_classifier)

    # %%
    from torch.optim import SGD, Optimizer
    from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR

    # if augmentation == 'mixup':
    #     LOSS_FUNCTION = augmentations.mixup_criterion()
    # else:
    LOSS_FUNCTION = nn.CrossEntropyLoss()

    n_epochs = 10
    scheduler_milestones = [10, 20]
    warmup_epochs = 3
    scheduler_gamma = 0.1
    learning_rate = 1e-6
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
        milestones=[5]
    )


    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    train_scheduler = seq_scheduler

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

    n_workers = 4
    print(model_name)
    for way in ways:
        for shot in shots:
            val_sampler = TaskSampler(
                val_set, n_way=way, n_shot=shot, n_query=n_query, n_tasks=n_validation_tasks
            )

            # Finally, the DataLoader. We customize the collate_fn so that batches are delivered
            # in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
            val_loader = DataLoader(
                val_set,
                batch_sampler=val_sampler,
                num_workers=n_workers,
                pin_memory=True,
                collate_fn=val_sampler.episodic_collate_fn,
            )

            accuracy = evaluate(few_shot_classifier, val_loader, device=DEVICE)
            print(f"Average accuracy : {(100 * accuracy):.2f} %")

    # %%
    # accuracy = evaluate(few_shot_classifier, val_loader, device=DEVICE)
    # print(f"Average accuracy : {(100 * accuracy):.2f} %")

    # %%
    # torch.save(few_shot_classifier, f'E:/Models/protonet_checkpoints/{model_name}.pt')


