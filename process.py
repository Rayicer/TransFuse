import numpy as np
import cv2
import os

root = 'data/' # change to your data folder path
data_f = ['ISIC-2017_Training_Data/', 'ISIC-2017_Validation_Data/', 'ISIC-2017_Test_v2_Data/']
mask_f = ['ISIC-2017_Training_Part1_GroundTruth/', 'ISIC-2017_Validation_Part1_GroundTruth/', 'ISIC-2017_Test_v2_Part1_GroundTruth/']
set_size = [2000, 150, 600]
save_name = ['train', 'val', 'test']

height = 192 # 384
width = 256 # 512

for j in range(3):

	print('processing ' + data_f[j] + '......')
	count = 0
	length = set_size[j]
	imgs = np.uint8(np.zeros([length, height, width, 3]))
	masks = np.uint8(np.zeros([length, height, width]))

	path = root + data_f[j]
	mask_p = root + mask_f[j]

	for i in os.listdir(path):
		if len(i.split('_'))==2:
			img = cv2.imread(path+i)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, (width, height))

			m_path = mask_p + i.replace('.jpg', '_segmentation.png')
			mask = cv2.imread(m_path, 0)
			mask = cv2.resize(mask, (width, height))

			imgs[count] = img
			masks[count] = mask

			count +=1 
			print(count)


	np.save('{}/data_{}.npy'.format(root, save_name[j]), imgs)
	np.save('{}/mask_{}.npy'.format(root, save_name[j]), masks)
