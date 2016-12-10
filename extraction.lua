require 'param'
require 'image'
require 'os'
torch.setdefaulttensortype('torch.FloatTensor')


local image_format = 'img'..'%06d_'

image_num = 1

function preprocess(img)
if img_type == 'yuv' then
return image.rgb2yuv(img)
else return img
  end
end
function ych(img)
if img_type == 'yuv' then 
  return img[{{1}}]
  else 
    img = image.rgb2yuv(img)[{{1}}]
return img

end
end
local rand = torch.range(1,50000)
function extractData(batch_size,patch_size,ch)

local ch = ch or 3
local data_fold = 'data/'
local img_iter = batch_size

if ch == 3 then 
  data_fold =  data_fold..rgb_fold
elseif ch ==1 then 
  data_fold =  Y_fold

else assert(nil,'wrong channel num')
end


local gt_mat = torch.Tensor(batch_size,ch,patch_size*scale,patch_size*scale)
local lr_mat = torch.Tensor(batch_size,ch,patch_size,patch_size)
--local ilr_mat = torch.Tensor(batch_size,ch,patch_size,patch_size)
local iter = 1 

while true do

while paths.filep(data_fold..gt_fold..string.format(image_format,rand[image_num])..'GT.png' ) ==false  
  do image_num = image_num + 1 end

local image_name = string.format(image_format,rand[image_num])



local GTname = data_fold .. gt_fold .. image_name..'GT.png'
local LRname = data_fold .. lr_fold .. image_name .. 'LR.png'
--local ILRname = data_fold.. ilr_fold .. image_name .. 'ILR.png'
local GT = image.load(GTname)
local LR = image.load(LRname)
--local ILR = image.load(ILRname)
assert(GT:size(2)==LR:size(2)*scale and GT:size(3)==LR:size(3)*scale,'size'..GT:size(2)..' ' ..LR:size(2)..' ' ..GT:size(3)..' ' ..LR:size(3))
math.randomseed(os.time())
local flip = math.random(2)
if flip == 1 then GT = image.hflip(GT) ; LR =image.hflip(LR) end

--GT = preprocess(GT)
--LR = preprocess(LR)

local h, w = LR:size(2), LR:size(3)
image_num =image_num+1
if image_num > 50000 then image_num =1;torch.manualSeed(os.time()); rand = torch.randperm(50000) end

for i = 1, img_iter do
if iter > batch_size then break; end
math.randomseed(sys.clock())
local crop_sx, crop_sy = math.random(0,w-patch_size), math.random(0,h-patch_size)
--print(image_name,crop_sx,crop_sy)
gt_mat[{{iter}}] = preprocess(image.crop(GT,crop_sx*scale,crop_sy*scale,crop_sx*scale+patch_size*scale,crop_sy*scale+patch_size*scale))
lr_mat[{{iter}}] = preprocess(image.crop(LR,crop_sx,crop_sy,crop_sx+patch_size,crop_sy+patch_size))
--image.save('conf/g'..image_num..'.jpg',gt_mat[{iter}])
--image.save('conf/l'..image_num..'.jpg',lr_mat[{iter}])

iter = iter + 1
if iter > batch_size then goto out; end

end


--print(image_num)


end
::out::


return gt_mat, lr_mat



end


