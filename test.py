import torch
import torchvision.utils
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import make_variable
from utils import get_data_loader, init_model, init_random_seed
from models import Discriminator, LeNetClassifier, LeNetEncoder
import params
import cv2
from PIL import Image
import numpy as np
from IPython.display import Image,display_png


def eval_target(encoder, classifier, data_loader):   
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        preds = classifier(encoder(images))
        preds = preds.to('cpu').detach().numpy().copy()[0]
        preds_idx = np.argmax(preds)
        preds_sort = preds.argsort()
        preds_first_idx = preds_sort[-1]
        preds_second_idx = preds_sort[-2]
        preds_third_idx = preds_sort[-3]

        pred_animal = classes[preds_first_idx]
        print('First : ', pred_animal)
        #print('Second : ', classes[preds_second_idx])
        #print('Third : ', classes[preds_third_idx])
        display_png(Image("datasets2/show_image/"+pred_animal+".png"))
        #img = Image.open("datasets2/show_image/"+pred_animal+".png")
        #img.show()

classes = ['cat', 'fox', 'gorilla', 'raccoon']

src_encoder_path = 'snapshots/ADDA-source-encoder-final.pt'
src_classifier_path = 'snapshots/ADDA-source-classifier-final.pt'
tgt_encoder_path = 'snapshots/ADDA-target-encoder-400.pt'

src_encoder = init_model(net=LeNetEncoder(),
                             restore=src_encoder_path)
src_classifier = init_model(net=LeNetClassifier(),
                            restore=src_classifier_path)
tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=tgt_encoder_path)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((100, 100)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
tgt_dataset2 = ImageFolder("datasets2/test_data", transform)
tgt_data_loader_eval = DataLoader(tgt_dataset2, batch_size=1, shuffle=True)
eval_target(tgt_encoder, src_classifier, tgt_data_loader_eval)
