import os

import torch
from tqdm import *

from config import parse_args
from data.data_helper import create_dataloaders
from models.model import MultiModal
from utils.util import setup_device, setup_seed, logout, build_optimizer, evaluate


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    loop_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
    with torch.no_grad():
        for index, batch in loop_val:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
            loop_val.set_description(f'Evaluation [{index}/{len(val_dataloader)}]')
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load the data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    # 3. training
    step = 0
    best_score = args.best_score

    # ema = EMA(model, 0.999)
    # ema.register()
    # fgm = FGM(model)
    for epoch in range(args.max_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for index, batch in loop:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            # model.train()
            # loss,accuracy,_,_ = model(batch)
            # loss = loss.mean()
            # accuracy = accuracy.mean()
            # loss.backward()
            # fgm.attack()
            # loss_sum,accuracy_sum,_,_  = model(batch)
            # loss_sum = loss_sum.mean()
            # accuracy_sum = accuracy.mean()
            # loss_sum.backward()
            # fgm.restore()
            #
            # optimizer.step()
            # optimizer.zero_grad()
            # scheduler.step()
            # ema.update()
            # model.zero_grad()
            # step += 1

            loop.set_description(f'Epoch [{epoch}/{args.max_epochs}]')
            loop.set_postfix(loss=loss.item(), acc=accuracy.item())

        # 4. validation
        # ema.apply_shadow()
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logout().info(f"Epoch {epoch}: loss {loss:.3f}, f1 {results['mean_f1']}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        # ema.restore()

def main():
    args = parse_args()  # input the parameters from config.py
    setup_device(args)  # decide the device
    setup_seed(args)  # set the random seed

    os.makedirs(args.savedmodel_path, exist_ok=True)  # mkdir in the system
    logout().info("Training/evaluation parameters: %s", args)  # print and save the log

    train_and_validate(args)  # train


if __name__ == '__main__':
    main()