require 'image'
require 'extraction'



torch.setdefaulttensortype('torch.FloatTensor')

valid_img_fold = 'data/test/BSD100/image_SRF_4/'

local data = {}
local batch_sz = 20
local patch_sz = 96

input = torch.Tensor(batch_sz,3,patch_sz,patch_sz)
target = torch.Tensor(batch_sz,3,patch_sz*scale,patch_sz*scale)

iter = 1

for file =1, 5 do

file = string.format('img_%03d_SRF_4_HR.png',file)
local gt = image.load(valid_img_fold..file)
local lr = image.scale(gt,gt:size(3)/scale,gt:size(2)/scale,'bicubic')
for i = 1, 4 do
local crop_sx, crop_sy = math.random(0,gt:size(3)/4-patch_sz), math.random(0,gt:size(2)/4-patch_sz)

input[{{iter}}] = preprocess(image.crop(lr,crop_sx,crop_sy,crop_sx+patch_sz,crop_sy+patch_sz))
target[{{iter}}] = preprocess(image.crop(gt,crop_sx*scale,crop_sy*scale,crop_sx*scale+patch_sz*scale,crop_sy*scale+patch_sz*scale))

iter = iter + 1

if iter > batch_sz then break end
end



end

data.input = input
data.target = target

torch.save('valid.t7',data)


