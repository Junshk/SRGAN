require 'image'
require 'extraction'



torch.setdefaulttensortype('torch.FloatTensor')

valid_img_fold = 'data/test/Set5/image_SRF_4/'

local data = {}
local batch_sz = 10
local patch_sz = patch_size

input = torch.Tensor(batch_sz,3,patch_sz,patch_sz)
target = torch.Tensor(batch_sz,3,patch_sz*scale,patch_sz*scale)

iter = 1
file_n = 1
while true do

file = string.format('img_%03d_SRF_4_HR.png',file_n)
local file_lr = string.format('img_%03d_SRF_4_LR.png',file_n)
local gt = image.load(valid_img_fold..file)
local lr = image.load(valid_img_fold..file_lr)

if gt:size(3)>=patch_sz*scale and gt:size(2)>=patch_sz*scale then
--local lr = image.scale(gt,gt:size(3)/scale,gt:size(2)/scale,'bicubic')

local crop_sx, crop_sy = math.random(0,gt:size(3)/4-patch_sz), math.random(0,gt:size(2)/4-patch_sz)

input[{{iter}}] = preprocess(image.crop(lr,crop_sx,crop_sy,crop_sx+patch_sz,crop_sy+patch_sz))
target[{{iter}}] = preprocess(image.crop(gt,crop_sx*scale,crop_sy*scale,crop_sx*scale+patch_sz*scale,crop_sy*scale+patch_sz*scale))

iter = iter + 1
else
file_n = file_n + 1 
end
if iter > batch_sz then break end


end

data.input = input
data.target = target

torch.save('valid.t7',data)


