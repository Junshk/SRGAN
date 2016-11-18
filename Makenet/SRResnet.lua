require 'cunn'
require 'cudnn'

----------------------------------
local function blocks(net)

local concat = nn.ConcatTable()
local base = nn.Sequential()

base:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
base:add(nn.BatchNormalization(64))
base:add(nn.ReLU(true))
base:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
base:add(nn.BatchNormalization(64))

concat:add(base)
concat:add(nn.Identity())

net:add(concat)
net:add(nn.CAddTable())
end

----------------------------
function SRResnet()

local net = nn.Sequential()
local block_num = 15
local C =3
local dconvK = 4
net:add(nn.SpatialConvolution(C,64,3,3,1,1,1,1))
net:add(nn.ReLU(true))

for iter = 1, block_num do
blocks(net)
end

-- DECONV & RELU
net:add(nn.SpatialFullConvolution(64,64,dconvK,dconvK,2,2,2,2,1,1))
net:add(nn.ReLU(true))
net:add(nn.SpatialFullConvolution(64,64,dconvK,dconvK,2,2,2,2,1,1))
net:add(nn.ReLU(true))

-- CONV
net:add(nn.SpatialConvolution(64,C,3,3,1,1,1,1))
return net 
end


