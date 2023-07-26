import os
import argparse
import torch
import data_setup, engine, model_builder

from torchvision import transforms

parse = argparse.ArgumentParser(description='Get some hyperparameters')

parse.add_argument(
    '--num_epochs',
    type=int,
    default=5,
    required=True
)

parse.add_argument(
    '--batch_size',
    type=int,
    default=32,
    required=True
)

parse.add_argument(
    '--hidden_units',
    type=int,
    default=10,
    required=True
)

parse.add_argument(
    '--lr',
    type=float,
    default=0.001
    required=True
)

args = parse.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.lr

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

model = model_builder.TinyVGG(
    in_features=3,
    hidden_units=HIDDEN_UNITS,
    out_features=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
