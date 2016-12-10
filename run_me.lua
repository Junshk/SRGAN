require 'param'

require 'Training'


require 'make_net'

model, criterion = make_net(nettype,loss_type)
losses ={}
testloss = {}
if paths.filep(netname..'.t7') == false then continue = false end
if continue == true then
losses = torch.load(netname..'trainerr.t7')
testloss =torch.load(netname..'testerr.t7')
model = torch.load(netname..'.t7')
end

if paths.filep('valid.t7') ==false then
dofile('valid.lua')
end
--assert(nil)






training(model, criterion, netname)




