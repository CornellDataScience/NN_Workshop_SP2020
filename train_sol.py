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


def calculate_iou_prec_recall(preds, label_masks, pred_threshold=0.0):
  """
  Calculate IoU, Precision and Recall per class for entire batch of images.
  Requires:
    preds: model preds array, shape        (batch, #c, h, w)
    label_masks: ground truth masks, shape (batch, #c, h, w)
    pred_threshold: Confidence threshold over which pixel prediction counted.
  Returns:
    ious, precs, recall per class: shape (#c)
  """
  # Change view so that shape is (batch, h, w, #c)
  preds = preds.transpose(0, 2, 3, 1)
  label_masks = label_masks.transpose(0, 2, 3, 1)

  # Reduce dimensions across all but classes dimension.
  preds = preds.reshape(-1, preds.shape[-1])
  label_masks = label_masks.reshape(-1, label_masks.shape[-1])

  preds = preds > pred_threshold
  intersection = np.logical_and(preds, label_masks)
  union = np.logical_or(preds, label_masks)
  iou_scores = np.sum(intersection, axis=0) / np.sum(union, axis=0)
  iou_scores[np.isnan(iou_scores)] = 0.0

  precision = np.sum(intersection, axis=0)/np.sum(preds, axis=0)
  precision[np.isnan(precision)] = 0.0

  recall = np.sum(intersection, axis=0)/np.sum(label_masks, axis=0)
  recall[np.isnan(recall)] = 0.0

  return iou_scores, precision, recall


def train_step(model, loss_fn, optimizer, images, labels):
  """
  Performs one training step over a batch.
  Passes the batch of images through the model, and backprops the gradients.
  Returns the resulting model predictions and loss values.
  """
  ## ===========================================================
  ## BEGIN: YOUR CODE.
  ## ===========================================================

  # Flush the gradient buffers
  optimizer.zero_grad()
  
  # Feed model
  preds = model(images)
  loss = loss_fn(preds, labels)

  # Backpropagate
  loss.backward()
  optimizer.step()

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
  
  # Feed model
  preds = model(images)
  loss = loss_fn(preds, labels)

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
  best_val_iou = -np.inf
  for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}:")

    ## Metrics 
    train_loss, val_loss = 0, 0
    train_iou, val_iou = 0, 0
    train_prec, val_prec = 0, 0
    train_recall, val_recall = 0, 0

    ## Training.
    model.train()
    for batch_index, (input_t, y) in enumerate(train_loader):
      # Shift to correct device
      input_t, y = input_t.to(device), y.to(device)

      preds, loss = train_step(model, loss_fn, optimizer, input_t, y)

      preds_arr = preds.detach().cpu().numpy()
      y_arr = y.detach().cpu().numpy()

      iou, prec, recall = calculate_iou_prec_recall(preds_arr, y_arr)

      train_loss += loss.item()
      train_iou += iou[0]
      train_prec += prec[0]
      train_recall += recall[0]
    
    # Get mean epoch-level metrics
    num_train_batches = batch_index + 1
    train_loss = train_loss/num_train_batches
    train_iou = train_iou/num_train_batches
    train_prec = train_prec/num_train_batches
    train_recall = train_recall/num_train_batches
    
    ## Validation
    model.eval()
    for batch_index, (input_t, y) in enumerate(val_loader):
      # Shift to correct device
      input_t, y = input_t.to(device), y.to(device)

      preds, loss = val_step(model, loss_fn, input_t, y)

      preds_arr = preds.detach().cpu().numpy()
      y_arr = y.detach().cpu().numpy()

      iou, prec, recall = calculate_iou_prec_recall(preds_arr, y_arr)

      val_loss += loss.item()
      val_iou += iou[0]
      val_prec += prec[0]
      val_recall += recall[0]

    # Get mean epoch-level metrics
    num_val_batches = batch_index + 1
    val_loss = val_loss/num_val_batches
    val_iou = val_iou/num_val_batches
    val_prec = val_prec/num_val_batches
    val_recall = val_recall/num_val_batches

    # Save model weights
    if val_iou > best_val_iou:
      best_val_iou = val_iou
      if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)
      print("Saving weights...")
      save_path = os.path.join(args.ckpt_path, "fcn_weights.bin")
      torch.save(model.state_dict(), save_path)

    ## Print a bunch of metrics:
    print(f"Training loss: {train_loss}")
    print(f"Training IoU: {train_iou}")
    print(f"Training prec: {train_prec}")
    print(f"Training recall: {train_recall}")
    print(f"Validation loss: {val_loss}")
    print(f"Validation recall: {val_recall}")
    print(f"Validation iou: {val_iou}")
    print(f"Validation prec: {val_prec}")
    print(f"Validation recall: {val_recall}")

    print(f"Finished epoch {epoch+1}.\n")
      
