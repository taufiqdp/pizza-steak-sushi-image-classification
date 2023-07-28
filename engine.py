import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device) -> Tuple[float, float]:

    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(
            torch.softmax(
                y_pred,
                dim=1),
            dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred_class)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device) -> Tuple[float, float]:

    test_loss, test_acc = 0, 0

    model.to(device)

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model(X_test)

            loss = loss_fn(test_pred, y_test)
            test_loss += loss.item()

            test_pred_class = test_pred.argmax(dim=1)
            test_acc += (test_pred_class==y_test).sum().item()/len(test_pred_class)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          writer,
          device: torch.device,
          progress: bool = True) -> Dict[str, list]:

    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    print(f'Running on {device} device.....')

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device)

        result['train_loss'].append(float(train_loss))
        result['train_acc'].append(train_acc)

        test_loss, test_acc = test_step(
            model,
            test_dataloader,
            loss_fn,
            device)

        result['test_loss'].append(float(test_loss))
        result['test_acc'].append(test_acc)

        if progress:
            print(
                f"┌Epoch: {epoch+1}\n"
                f"└──train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        writer.add_scalars(
            main_tag='Loss',
            tag_scalar_dict={
                'train_loss': train_loss,
                'test_loss': test_loss
            },
            global_step=epoch
        )

        writer.add_scalars(
            main_tag='Accuracy',
            tag_scalar_dict={
                'train_acc': train_acc,
                'test_acc': test_acc
            },
            global_step=epoch
        )

        writer.add_graph(
            model=model,
            input_to_model=torch.randn(32, 3, 224, 224).to(device)
        )

    writer.close()

    return result


def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str = None):

    timestamp = datetime.now().strftime('%Y-%m-%d')
    if extra:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir)
