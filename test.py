import torch
import torchvision.utils
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import make_variable
from utils import get_data_loader, init_model, init_random_seed
from models import Discriminator, LeNetClassifier, LeNetEncoder
import params

def eval_target(encoder, classifier, data_loader):
    #encoder.eval()
    #classifier.eval()
    res = 0

    #classes = ['cat', 'fox', 'gorilla', 'raccoon']
    for (images, labels) in data_loader:
        #imshow(torchvision.utils.make_grid(images))
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        print(labels)
        preds = classifier(encoder(images))
        _, predicted = torch.max(preds, 1)
        print(predicted)
        if labels == predicted:
            res += 1
        print("---------")

    print('acc', res / len(data_loader)

src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
src_classifier = init_model(net=LeNetClassifier(),
                            restore=params.src_classifier_restore)
tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((100, 100)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
tgt_dataset = ImageFolder("datasets2/human_dataset/resized", transform) # ここは適当にラベル付きのデータを用意する

train_ratio = 0.8
tgt_train_size = int(train_ratio * len(tgt_dataset))
tgt_val_size  = len(tgt_dataset) - tgt_train_size   
tgt_data_size  = {"train":tgt_train_size, "val":tgt_val_size}
print(tgt_data_size)

tgt_data_train, tgt_data_val = random_split(tgt_dataset, [tgt_train_size, tgt_val_size])
tgt_data_loader = DataLoader(tgt_data_train, batch_size=1, shuffle=True)
tgt_data_loader_eval = DataLoader(tgt_data_val, batch_size=1, shuffle=True)
eval_target(src_encoder, src_classifier, tgt_data_loader_eval)
eval_target(tgt_encoder, src_classifier, tgt_data_loader_eval)
