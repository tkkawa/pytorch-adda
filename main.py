import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    #src_data_loader = get_data_loader(params.src_dataset)
    #src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    #tgt_data_loader = get_data_loader(params.tgt_dataset)
    #tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # Dataset を作成する。
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((100, 100)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    src_dataset = ImageFolder("datasets2/animal_face_dataset/resized", transform)
    tgt_dataset = ImageFolder("datasets2/human_dataset/resized", transform)
    train_ratio = 0.8
    src_train_size = int(train_ratio * len(src_dataset))
    src_val_size  = len(src_dataset) - src_train_size   
    src_data_size  = {"train":src_train_size, "val":src_val_size}
    print(src_data_size)
    tgt_train_size = int(train_ratio * len(tgt_dataset))
    tgt_val_size  = len(tgt_dataset) - tgt_train_size   
    tgt_data_size  = {"train":tgt_train_size, "val":tgt_val_size}
    print(tgt_data_size)
    
    src_data_train, src_data_val = random_split(src_dataset, [src_train_size, src_val_size])
    src_data_loader = DataLoader(src_data_train, batch_size=3, shuffle=True)
    src_data_loader_eval = DataLoader(src_data_val, batch_size=2, shuffle=True)

    tgt_data_train, tgt_data_val = random_split(tgt_dataset, [tgt_train_size, tgt_val_size])
    tgt_data_loader = DataLoader(tgt_data_train, batch_size=2, shuffle=True)
    tgt_data_loader_eval = DataLoader(tgt_data_val, batch_size=, shuffle=True)
    #print(type(dataloader))
    print(type(src_data_loader))
    print(type(tgt_data_loader))
    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, src_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
