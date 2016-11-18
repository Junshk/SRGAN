require 'param'



nettype = 'ResMSE'
netname = nettype 
require 'make_net'
model, criterion = make_net(nettype)




if paths.filep('valid.t7') ==false then
dofile('valid.lua')
end
assert(nil)
require 'Training'

Training(model, criterion, netname)




