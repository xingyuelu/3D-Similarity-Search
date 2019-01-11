
import logging
import random
import numpy as np
import scipy.io
from path import Path
import argparse

#import voxnet
from voxnet import npytar

from voxnet.data import shapenet10

def write(records, fname):
    writer = npytar.NpyTarWriter(fname)
    for (classname, instance, rot, fname) in records:
        class_id = int(shapenet10.class_name_to_id[classname])
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        arrpad = np.zeros((32,)*3, dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        writer.add(arrpad, name)
    writer.close()

def writeNPZ(train_set, test_set, fname):
    # prepare Xtr, Ytr, Xte, Yte
    Xtr = []
    Ytr = []
    Xte = []
    Yte = []
    for (classname, instance, rot, fname) in train_set:
        class_id = int(shapenet10.class_name_to_id[classname])
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        arrpad = np.zeros((32,)*3, dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        Xtr.append(arrpad)
        Ytr.append(class_id)  

    for (classname, instance, rot, fname) in test_set:
        class_id = int(shapenet10.class_name_to_id[classname])
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        arrpad = np.zeros((32,)*3, dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        Xte.append(arrpad)
        Yte.append(class_id)  
    #data['X_train']= Xtr
    #data['y_train']= Ytr
    #data['X_test'] = Xte
    #data['y_test'] = Yte
    #print(Ytr)
    #print(np.shape(Ytr))
    Ytr = np.array(Ytr)
    Xtr = np.array(Xtr)
    Yte = np.array(Yte)
    Xte = np.array(Xte)
    #np.savez_compressed(fname, a=Ytr)
    test_array = np.random.rand(3, 2) 
    test_vector = np.random.rand(4)
    print(fname)
    np.savez_compressed('shape', a=Xtr, b=Ytr, c=Xte, d= Yte)

    print('saved')

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')

#base_dir = Path('~/code/3DShapeNets2/3DShapeNets/volumetric_data').expand()
base_dir = (args.data_dir/'volumetric_data').expand()

records = {'train': [], 'test': []}

logging.info('Loading .mat files')
for fname in sorted(base_dir.walkfiles('*.mat')):
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'):
        continue
    elts = fname.splitall()
    instance_rot = Path(elts[-1]).stripext()
    instance = instance_rot[:instance_rot.rfind('_')]
    rot = int(instance_rot[instance_rot.rfind('_')+1:])
    split = elts[-2]
    classname = elts[-4].strip()
    if classname not in shapenet10.class_names:
        continue
    records[split].append((classname, instance, rot, fname))


# just shuffle train set
#logging.info('Saving train npy tar file')
train_records = records['train']
random.shuffle(train_records)
#write(train_records, 'shapenet10_train.tar')

# order test set by instance and orientation
#logging.info('Saving test npy tar file')
test_records = records['test']
test_records = sorted(test_records, key=lambda x: x[2])
test_records = sorted(test_records, key=lambda x: x[1])
#write(test_records, 'shapenet10_test.tar')
logging.info('Saving npz file')
writeNPZ(train_records,test_records,'shape')
