import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fcn import FCN, load_fcn
from dataset import get_data_loaders


def train_step(model, loss_fn, optimizer, images, labels):
  """
  Performs one training step over a batch.
  Passes the batch of images through the model, and backprops the gradients.
  Returns the resulting model predictions and loss values.
  """
  ## ===========================================================
  ## BEGIN: YOUR CODE.
  ## ===========================================================
  preds, loss = None, None


  ## ===========================================================
  ## END: YOUR CODE.
  ## ===========================================================
  return preds, loss

def val_step(model, loss_fn, images, labels):
  """
  Performs one validation step over a batch.
  Passes the batch of images through the model.
  Returns the resulting model predictions and loss values.
  """
  ## ===========================================================
  ## BEGIN: YOUR CODE.
  ## ===========================================================
  preds, loss = None, None

  ## ===========================================================
  ## END: YOUR CODE.
  ## ===========================================================

  return preds, loss


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to train segmentation models.")
  parser.add_argument("--id_path",
                      type=str,
                      required=True,
                      help="Path to file containing image ids.")
  parser.add_argument("--im_path",
                      type=str,
                      required=True,
                      help="Path to directory containing images.")
  parser.add_argument("--label_path",
                      type=str,
                      required=True,
                      help="Path to directory containing labels")
  parser.add_argument("--config",
                      type=str,
                       required=True,
                       help="Path to .json training hyper parameter config file.")
  parser.add_argument("--ckpt_path",
                      type=str,
                      default=os.path.join('.', 'weights'),
                      help="Path to directory to save model weights.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()

  # Get training config file
  with open(args.config) as f:
    config = json.load(f)

  ## IGNORE: This is for me only for training on gpu. Will remove later.
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")

  # Load model.
  model = load_fcn(num_classes=1)
  model.to(device)

  # Load optimizer and loss
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=config["lr"])

  ## Set up Data Loaders.
  epochs = config["epochs"]
  train_loader, val_loader, _ = get_data_loaders(
    args.id_path,
    args.im_path,
    args.label_path,
    batch_size=config["batch_size"]
  ) 

  ## Begin training
  best_val_loss = np.inf
  for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}:")

    ## Metrics 
    train_loss, val_loss = 0, 0

    ## Training.
    model.train()
    for batch_index, (input_t, y) in enumerate(train_loader):
      # Shift to correct device
      input_t, y = input_t.to(device), y.to(device)

      preds, loss = train_step(model, loss_fn, optimizer, input_t, y)

      train_loss += loss.item()
    
    # Get mean epoch-level metrics
    num_train_batches = batch_index + 1
    train_loss = train_loss/num_train_batches
    
    ## Validation
    model.eval()
    for batch_index, (input_t, y) in enumerate(val_loader):
      # Shift to correct device
      input_t, y = input_t.to(device), y.to(device)

      preds, loss = val_step(model, loss_fn, input_t, y)

      val_loss += loss.item()

    # Get mean epoch-level metrics
    num_val_batches = batch_index + 1
    val_loss = val_loss/num_val_batches

    # Save model weights
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)
      print("Saving weights...")
      save_path = os.path.join(args.ckpt_path, "fcn_weights.bin")
      torch.save(model.state_dict(), save_path)

    ## Print metrics:
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")

    print(f"Finished epoch {epoch+1}.\n")
      
