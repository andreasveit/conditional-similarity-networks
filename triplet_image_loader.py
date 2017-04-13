from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

filenames = {'train': ['class_tripletlist_train.txt', 'closure_tripletlist_train.txt', 
                'gender_tripletlist_train.txt', 'heel_tripletlist_train.txt'],
             'val': ['class_tripletlist_val.txt', 'closure_tripletlist_val.txt', 
                'gender_tripletlist_val.txt', 'heel_tripletlist_val.txt'],
             'test': ['class_tripletlist_test.txt', 'closure_tripletlist_test.txt', 
                'gender_tripletlist_test.txt', 'heel_tripletlist_test.txt']}

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, filenames_filename, conditions, split, n_triplets, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.root = root
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        if split == 'train':
            fnames = filenames['train']
        elif split == 'val':
            fnames = filenames['val']
        else:
            fnames = filenames['test']
        for condition in conditions:
            for line in open(os.path.join(self.root, 'tripletlists', fnames[condition])):
                triplets.append((line.split()[0], line.split()[1], line.split()[2], condition)) # anchor, far, close   
        # print(triplets[:100])   
        np.random.shuffle(triplets)
        # print(triplets[:100])  
        self.triplets = triplets[:int(n_triplets * 1.0 * len(conditions) / 4)]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3, c = self.triplets[index]
        if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
            img1 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]))
            img2 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]))
            img3 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, c
        else:
            return None

    def __len__(self):
        return len(self.triplets)