import os.path as osp
import openslide
from pathlib import Path

#BASE_TRUTH_DIR = '/home/wli/Downloads/CAMELYON16/masking'

slide_path = '/home/wli/Downloads/googlepred/tumor_026.tif'
#truth_path = osp.join(BASE_TRUTH_DIR, 'tumor_009_mask.tif')

slide = openslide.open_slide(slide_path)
m = slide.read_region((56000, 123200), 0, (224, 224))
m = np.array(m)
m = m[:,:,:3]
# adding color noise based on the rgb image patches
m_colornoise = m + np.random.uniform(0, 20, size=(1,3))
m_integ = m_colornoise.astype('uint8')
# adding color noise based on the HSV image patches
import cv2
m_hsv = cv2.cvtColor(m, cv2.COLOR_BGR2HSV)
m_hsv_colornoise = m_hsv + np.random.uniform(0, 20, size=(1,3))
m_hsv_colornoise_integ = m_hsv_colornoise.astype('uint8')
m_rgb = cv2.cvtColor(m_hsv_colornoise_integ, cv2.COLOR_HSV2BGR)

#visualize them in a figure
f, axes = plt.subplots(1, 3, figsize=(30, 10));
ax = axes.ravel()
ax[0].imshow(m);
ax[0].set_title('before adding color noise')
ax[1].imshow(m_colornoise.astype('uint8'));
ax[1].set_title('after adding color noise on RGB image patch')
ax[2].imshow(m_rgb)
ax[2].set_title('after adding color noise on HSV image patch')
