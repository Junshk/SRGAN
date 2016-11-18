require 'nn'
require 'Makenet/SRResnet'
require 'Makenet/VGG54'
require 'Makenet/SRGAN'


function make_net(option)

local net, criterion

if type(option) ~= 'table' then assert(nil, 'wrong net option') end

  if option.type == 'ResMSE' then
    net = SRResnet()
    criterion = nn.MSECriterion()
  elseif option.type == 'ResVGG' then
    net = SRResnet()
    criterion = VGG54()
  elseif option.type == 'GANMSE' then
    net = SRGAN()
    criterion = nn.MSECriterion()
  elseif option.type == 'GANVGG' then
    net = SRGAN()
    criterion = VGG54()
  else assert(nil, 'wrong net type') end
return net, criterion 
end

