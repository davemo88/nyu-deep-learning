require 'nn'
require 'cunn'

require('batchflip.lua')

dofile('models/unsupervised_model_simple.lua')

file = 'trained/cae_simple6.t7b'
depth  = 0
mod = torch.load(file)
local m = mod
while true do
  depth = depth + 1
  if (m:size() == 2) then
    break 
  end
  m = m:get(2)
end
nInputPlanes = 64* math.pow(2,depth-1)
nOutputPlanes = 64*math.pow(2,depth)
new_layer = nn.Sequential()
pooling, encoder = Encoder(nInputPlanes,nOutputPlanes)
new_layer:add(encoder)
new_layer:add(Decoder(nOutputPlanes,nInputPlanes, pooling))
current = mod
for i = 1,depth-1 do
	current = current:get(2)
end	
current:insert(new_layer, 2)

return mod


