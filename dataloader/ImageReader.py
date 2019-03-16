import os
import pickle
import torch
import torch.utils.data as data
import torchvision.utils as vutils
from dataloader.cvtransforms import *


def batch_collate_fn(batch):
    H_list = [x[0].shape[-2] for x in batch]
    W_list = [x[0].shape[-1] for x in batch]
    H_max = max(H_list)
    W_max = max(W_list)
    images = []
    labels = []
    for (img, label) in batch:
        padding = torch.nn.ZeroPad2d([0, W_max-img.shape[-1], 0, H_max-img.shape[-2]])

        tmp_img = padding(img)
        tmp_label = padding(label)

        images.append(tmp_img)
        labels.append(tmp_label)
    return torch.stack(images), torch.stack(labels)




class ImageFolder(data.Dataset):
    __slots__ = ['root_path', 'source_path', 'target_path', 'len', 'transforms']

    def __init__(self, root_path, target='train'):
        super(ImageFolder, self).__init__()
        self.root_path = root_path
        self.target = target
        self.source_path, self.target_path = self.get_images_path(self.root_path)
        self.len = len(self.source_path)

        self.transforms = self.transforms_init()

    def __getitem__(self, index):
        source_image_path = self.source_path[index]
        target_image_path = self.target_path[index]
        source_image = self.image_loader(source_image_path)
        target_image = self.image_loader(target_image_path)

        source_image, target_image = self.transforms([source_image, target_image])
        return source_image, target_image

    def __len__(self):
        return self.len

    def get_images_path(self, root: str):
        if self.target.lower() == 'train':
            imglist_file = open(os.path.join(root, 'imglist_train.pkl'), 'rb')
        elif self.target.lower() == 'validation':
            imglist_file = open(os.path.join(root, 'imglist_validation.pkl'), 'rb')
        else:
            imglist_file = open(os.path.join(root, 'imglist_test.pkl'), 'rb')

        imglist = pickle.load(imglist_file)
        source_list = [os.path.join(root, 'SynthText', i) for i in imglist]
        target_list = [os.path.join(root, 'Synth_Text_target', i) for i in imglist]

        return source_list, target_list

    @staticmethod
    def image_loader(path):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return None

    def transforms_init(self):
        if self.target.lower() == 'train':
            return Compose([
                RandomOrderAndApply([ColorJitter(0.2, 0.2, 0.2, 0.05),
                                     RandomPerspective(),
                                     RandomChoice((SPNoise(), GaussianNoise()))
                                     ]),
                # Resize((400, 600)),
                ToTensor(),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.target.lower() == 'test' or self.target.lower() == 'validation':
            return Compose([
                # Resize((400, 600)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


def create_dataset(opt):
    image_train_dataset = ImageFolder(root_path=opt.dataroot, target='train')
    train_loader = data.DataLoader(image_train_dataset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=2)

    image_validation_dataset = ImageFolder(root_path=opt.dataroot, target='validation')
    validation_loader = data.DataLoader(image_validation_dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=2)

    return train_loader, validation_loader


if __name__ == '__main__':
    manualSeed = 2128
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    dataset_path = 'F:/workshop/dataset'
    image_dataset = ImageFolder(root_path=dataset_path)
    dataloders = data.DataLoader(image_dataset, batch_size=1, shuffle=True, num_workers=2)
    total = len(image_dataset)
    cv2.namedWindow('0', 0)
    for idx, (img_source, img_target) in enumerate(dataloders):

        if idx % 1 == 0:
            print('{} of {} is processed'.format(idx, total))
            img_pair = vutils.make_grid(torch.cat([img_source,
                                                   img_target,
                                                   torch.abs(img_target - img_source)],
                                                  dim=0), nrow=3)
            cv2.imshow('0', img_pair.numpy().transpose([1, 2, 0])[:, :, ::-1])
            cv2.waitKey(1)
