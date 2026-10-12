import random
from pathlib import Path
from statistics import mean
import cProfile
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
import copy
import torchvision.transforms as transforms
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
import timm
from pytorch_datasets import ISIC2018, BreakHis, Papsmear
from pytorch_datasets import augmentations

from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
DEVICE = "cuda"

num_meta_iterations = 50

train_batch_size    = 10
eval_batch_size     = 20
def mem():
    print(torch.cuda.memory_allocated() / 1000000)

def reptile(models, dataset, n_way, mag=100):
    print(mag)
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    cutout_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        augmentations.Cutout(n_holes=1, length=64),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZATION),
    ])
    simple_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # %%
    n_way   = n_way
    n_eval_shot  = 3
    n_train_shot  = 10
    n_query = 10

    n_episodes          = num_meta_iterations * train_batch_size
    n_eval_episodes     = num_meta_iterations * eval_batch_size

    DEVICE = "cuda"
    n_workers = 6

    # Selecting augmentation
    augments = [None, 'cutout', 'mixup', 'cutmix']
    augmentation  = augments[0]

    if augmentation == 'cutout':
        transform = cutout_transform
    else:
        transform = None

    datasets  = {0: 'ISIC2018', 1: 'BreakHis', 2: 'Papsmear'}
    dataset   = datasets[dataset] # Dataset ID

    if dataset == 'ISIC2018':
        root  = 'C:/Datasets/ISIC2018'
        train_set = ISIC2018(
            root=root, split='train', 
            training=True, 
            # transforms=transform
        )
        val_set   = ISIC2018(
            root=root, split='validation', 
            training=False,
        )

    elif dataset == 'BreakHis':
        root          = 'C:/Datasets/BreaKHis_v1'
        magnification = mag

        train_set = BreakHis(
            root=root, magnification=magnification, split='train', 
            training = True,
            # transforms=transform
        )
        val_set   = BreakHis(
            root=root, magnification=magnification, split='validation', 
            training=False,
        )

    elif dataset == 'Papsmear':
        train_root = r"C:\Datasets\smear2005\New database pictures\train"
        test_root  = r"C:\Datasets\smear2005\New database pictures\test"

        train_set = Papsmear(root=train_root, training=True,
                            #  transform=transform
                            )
        val_set   = Papsmear(root=test_root, training=False,)

    train_sampler = TaskSampler(
        train_set, 
        n_way=n_way, 
        n_shot=n_train_shot, 
        n_query=n_query, 
        n_tasks=n_episodes
    )
    val_sampler = TaskSampler(
        val_set, 
        n_way=n_way, 
        n_shot=n_eval_shot, 
        n_query=n_query, 
        n_tasks=n_eval_episodes
    )

    # Finally, the DataLoader. We customize the collate_fn so that batches are delivered
    # in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        # persistent_workers=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    for model_name in models:
        train(model_name, dataset, train_loader, val_loader, val_set, augmentation, n_way, mag)

def train(model_name, dataset, train_loader, val_loader, val_set, augmentation, n_way, mag):
    n_query = 10

    # train_batch_size    = 10
    # eval_batch_size     = 20
    # num_meta_iterations = 40
    num_inner_loop      = 10
    num_inner_loop_eval = 10
    meta_step_size = 1
    inner_loop_lr = 0.01

    model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=n_way)

    model = model
    # model = model.to(DEVICE)



    criterion = nn.CrossEntropyLoss()

    n_epochs = 100
    scheduler_milestones = [120, 160]
    scheduler_gamma = 0.1
    learning_rate = 1e-5
    tb_logs_dir = Path(".")

    train_optimizer = SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=5e-4,
    )

    warmup_scheduler = LinearLR(
        train_optimizer,
        start_factor=0.02,
        end_factor=1, 
        total_iters=5
    )
    cosine_scheduler = CosineAnnealingLR(
        train_optimizer,
        T_max=n_epochs-5,
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
    tb_write = False

    train_dataset_iterator  = iter(train_loader)
    val_dataset_iterator    = iter(val_loader)

    if tb_write:
        writer = SummaryWriter(comment=f'reptile_{model_name}_{dataset}_inner{num_inner_loop}')
    with tqdm(range(num_meta_iterations), position=0, desc="Outer loop", leave=False) as reptile_tqdm:
        for meta_iter in reptile_tqdm:
            
            model.train()
            
            task_models = []

            for _ in range(train_batch_size):
                # print(_)
                support_images, support_labels, query_images, query_labels, _ = next(train_dataset_iterator)
            
                task_model = copy.deepcopy(model).to(DEVICE)
                task_optimizer = optim.SGD(task_model.parameters(), lr=inner_loop_lr)

                for inner_iter in range(num_inner_loop):
                    # print(inner_iter)
                    data, target = support_images.to(DEVICE), support_labels.to(DEVICE)
                    
                    task_optimizer.zero_grad()

                    preds = task_model(data)

                    loss = criterion(preds, target)
                    
                    loss.backward()
                    
                    task_optimizer.step()

                task_models.append(task_model.cpu())
            # print('m')
            for model_param, *task_model_params in zip(model.parameters(), *(task_model.parameters() for task_model in task_models)):
                model_param.data = model_param.data + (meta_step_size * (sum(task_model_param.data for task_model_param in task_model_params) / train_batch_size - model_param.data))
            # print('end')
            
            # for task_model in task_models:
            #     del task_model
            task_models = []
            # Evaluation
            if meta_iter % 50 == 100:
                model.eval()
                total_accuracy = 0
                num_eval_tasks = eval_batch_size

                for _ in range(eval_batch_size):
                    support_images, support_labels, query_images, query_labels, _ = next(val_dataset_iterator)
                    task_model = copy.deepcopy(model).to(DEVICE)
                    task_optimizer = optim.SGD(task_model.parameters(), lr=inner_loop_lr)

                    for inner_iter in range(num_inner_loop_eval):
                        data, target = support_images.to(DEVICE), support_labels.to(DEVICE)

                        task_optimizer.zero_grad()

                        preds = task_model(data)

                        loss = criterion(preds, target)
                        loss.backward()

                        task_optimizer.step()

                    # Test the adapted model on the query set
                    data, target = query_images.to(DEVICE), query_labels.to(DEVICE)
                    preds = task_model(data)
                    pred_labels = preds.argmax(dim=1)
                    accuracy = (pred_labels == target).float().mean().item()
                    total_accuracy += accuracy
                mean_accuracy = total_accuracy / num_eval_tasks
                if tb_write:
                    writer.add_scalar("acc", mean_accuracy, meta_iter)
                # print(f"Meta-Iteration {meta_iter}: Validation accuracy: {mean_accuracy:.4f}")
                # reptile_tqdm.set_postfix(test_accuracy=mean_accuracy)
    del task_models
    torch.cuda.empty_cache()
    test_batch_size = 100
    n_shots = [3,  5, 10]
    # n_shots = [5, 10]

    print(model_name)
    torch.save(model, f'{model_name}_mag{mag}_way{n_way}.pth')
    # for number_shot in n_shots:
    #     test_sampler = TaskSampler(
    #         val_set, n_way=n_way, n_shot=number_shot, n_query=5, n_tasks=test_batch_size
    #     )

    #     test_loader = DataLoader(
    #         val_set,
    #         batch_sampler=test_sampler,
    #         num_workers=6,
    #         pin_memory=True,
    #         collate_fn=test_sampler.episodic_collate_fn,
    #     )

    #     model.eval()
    #     total_accuracy = 0
    #     num_eval_tasks = len(test_loader)

    #     with tqdm(
    #             enumerate(test_loader), total=len(test_loader), desc="eval loop", position=1, leave=False,
    #         ) as tqdm_val:
    #             for episode_index, (
    #                 support_images,
    #                 support_labels,
    #                 query_images,
    #                 query_labels,
    #                 _,
    #             ) in tqdm_val:

    #                 task_model = copy.deepcopy(model).to(DEVICE)
    #                 task_optimizer = optim.SGD(task_model.parameters(), lr=inner_loop_lr)

    #                 for inner_iter in range(num_inner_loop_eval):
    #                     data, target = support_images.to(DEVICE), support_labels.to(DEVICE)

    #                     task_optimizer.zero_grad()
                        
    #                     preds = task_model(data)

    #                     loss = criterion(preds, target)
    #                     loss.backward()

    #                     task_optimizer.step()

    #                 # Test the adapted model on the query set
    #                 data, target = query_images.to(DEVICE), query_labels.to(DEVICE)
    #                 preds = task_model(data)
    #                 pred_labels = preds.argmax(dim=1)
    #                 accuracy = (pred_labels == target).float().mean().item()
    #                 total_accuracy += accuracy
    #                 del preds
    #     mean_accuracy = total_accuracy / num_eval_tasks
    #     print(f"{model_name}-{augmentation} {n_way}-way {number_shot}-shot Test accuracy: {mean_accuracy:.4f}")
    #     # reptile_tqdm.set_postfix(test_accuracy=mean_accuracy)


