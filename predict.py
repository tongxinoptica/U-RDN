import torch
from torchvision import transforms
from model import RDR_model
from unit import to_psnr, to_ssim_skimage, to_ssim, to_pearson
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

Tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
RDR_model = RDR_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
RDR_model.to(device)
print("model ready")

Image_path = './images_predict/f/in_celeb.jpg'  # You can select other images, but they should match the parameter files
label_path = './images_predict/gt_celeb.jpg'
img = Image.open(Image_path)
img = Tensor(img)
img = torch.reshape(img, (1, 1, 256, 256))
label = Image.open(label_path)
label = Tensor(label)
label = torch.reshape(label, (1, 1, 256, 256)) 
RDR_model.load_state_dict(torch.load('./parameter/mse_gr36_f')) # Select the correct parameter file
b = torch.ones(256, 256).to(device)

RDR_model.eval()
with torch.no_grad():
    img = img.to(device)
    label = label.to(device)
    output = RDR_model(img)
    output = torch.where(output < 1, output, b)
    # print(torch.max(output))
    mseloss = nn.MSELoss()
    mse = mseloss(output, label)
    psnr = to_psnr(output, label)
    npcc = to_pearson(output, label)
    print(psnr)
    print(mse)
    print(npcc)
    ssim = to_ssim_skimage(output, label)
    ssim1 = to_ssim(output, label)
    print(ssim)
    #print(ssim1)
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output[0])  
    plt.imshow(output_pil, cmap='gray')
    plt.axis('off')
    # plt.savefig('./out_f_celeb.jpg', bbox_inches='tight', pad_inches=0) # Save output
    plt.show()
