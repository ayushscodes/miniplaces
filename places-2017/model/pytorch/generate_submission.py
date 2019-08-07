# Submission format: <filename> <label(1)> <label(2)> <label(3)> <label(4)> <label(5)>
# Example line: val/00000005.jpg 65 3 84 93 67

# Your team, LaCroixNet, has been registered for the challenge.
# Your team code (case-sensitive) for submitting to the leaderboard is: fFR7jWuG2XqiAImnJQrN
from __future__ import print_function, division
import os, sys, time

sys.path.append('../')

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from miniplaces_dataset import MiniPlacesTestSet

import vgg_pytorch as VGG

def main():
  # Apply same transforms to the test set as the training set, except without randomized cropping and flipping.
  transform = transforms.Compose(
      [transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize((0.45834960097, 0.44674252445, 0.41352266842), (0.229, 0.224, 0.225))])

  # Set up a test loader, which outputs image / filename pairs.
  test_set = MiniPlacesTestSet('/home/milo/envs/tensorflow35/miniplaces/data/images/test/',
            transform=transform, outfile=str(int(time.time())) + 'predictions.txt')

  # Define the model.
  # Not using CUDA for now, so that we can run this while training.
  model = VGG.vgg11(num_classes=100)
  model.features = torch.nn.DataParallel(model.features)
  model.cuda()

  checkpoint_file = './model_best.pth.tar'

  # If checkpoint file is given, resume from there.
  if os.path.isfile(checkpoint_file):
    print("Loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict']) # Get frozen weights.
    print("Loaded checkpoint '{}' (epoch {}, best_prc1 {})".format(checkpoint_file, start_epoch, best_prec1))
  else:
    print("No checkpoint found at {}".format(checkpoint_file))
    assert(False)

  model.eval() # Set to eval mode to prevent dropout from occurring.

  for i, data in enumerate(test_set):
    image, filename = data
    image = image.unsqueeze(0)
    inputs_var = torch.autograd.Variable(image)

    predictions = model(inputs_var)
    _, top5 = predictions.topk(5, 1, True, True)
    top5 = top5.t()

    labels = [top5.data[i][0] for i in range(5)]

    # Write the top 5 labels as a new line.
    test_set.write_labels(filename, labels)

    if i % 100 == 0:
      print('Processed %d / %d' % (i, len(test_set)))

  print('Finished preparing submission!')

if __name__ == '__main__':
  main()