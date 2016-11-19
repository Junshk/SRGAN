require 'param'

torch.setdefaulttensortype('torch.FloatTensor')


local image_format = 'img'..'%06d_'

image_num = 1

function preprocess(img)
return image.rgb2yuv(img)
end

function extractData(batch_size,patch_size,ch)

local ch = ch or 3
local data_fold = fold
local img_iter = 1--batch_size


if ch == 3 then data_fold = data_fold .. rgb_fold
elseif ch ==1 then data_fold = data_fold .. Y_fold
else assert(nil,'wrong channel num')
end


local gt_mat = torch.Tensor(batch_size,ch,patch_size*scale,patch_size*scale)
local lr_mat = torch.Tensor(batch_size,ch,patch_size,patch_size)
--local ilr_mat = torch.Tensor(batch_size,ch,patch_size,patch_size)
local iter = 1 

while true do

local image_name = string.format(image_format,image_num)

local GTname = data_fold .. gt_fold .. image_name..'GT.png'
local LRname = data_fold .. lr_fold .. image_name .. 'LR.png'
--local ILRname = data_fold.. ilr_fold .. image_name .. 'ILR.png'

local GT = image.load(GTname)
local LR = image.load(LRname)
--local ILR = image.load(ILRname)

--GT = preprocess(GT)
--LR = preprocess(LR)

local h, w = LR:size(2), LR:size(3)

for i = 1, img_iter do
if math.random(2) == 1 then GT = image.hflip(GT) ; LR =image.hflip(LR) end
local crop_sx, crop_sy = math.random(0,w-patch_size), math.random(0,h-patch_size)
gt_mat[{{iter}}] = preprocess(image.crop(GT,crop_sx*scale,crop_sy*scale,crop_sx*scale+patch_size*scale,crop_sy*scale+patch_size*scale))
lr_mat[{{iter}}] = preprocess(image.crop(LR,crop_sx,crop_sy,crop_sx+patch_size,crop_sy+patch_size))
iter = iter + 1

end

img_iter = img_iter +1
if iteration > batch_size then break; end

end


if img_iter > 50000 then img_iter =1 end


return gt_mat, lr_mat



end


