require 'cunn'
require 'cudnn'


local function blocks(net)

local concat = nn.ConcatTable()
local base = nn.Sequential()

base:add(nn.SpatialConvolution())
base:add(nn.BatchNormalization())
base:add(nn.ReLU())
base:add(nn.SpatialConvolution())
base:add(nn.BatchNormalization())

concat:add(base)
concat:add(nn.Identity())

net:add(concat)
net:add(nn.CAddTable())
end
----------------------------
function make_srresnet()

local net = nn.Sequential()

net:add(nn.SpatialConvolution())
net:add(nn.ReLU())

for iter = 1, block_num do
blocks(net)
end

-- DECONV & RELU

-- CONV

return net 
end


