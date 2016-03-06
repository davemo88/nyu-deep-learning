require 'nn'
require 'image'
require 'optim'
require 'cunn'


mod_old = torch.load('trained/swwae_simple110.t7b'):get(3)
mod_new = nn.Sequential()
for i = 1,4 do
	mod_new:add(mod_old:get(i))
end	
maxpool1 = mod_new:get(4)
mod_new:add(nn.SpatialBatchNormalization(64))
mod_new:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
mod_new:add(nn.Tanh())
mod_new:add(nn.Dropout(0.4))
maxpool2 = nn.SpatialMaxPooling(2,2,2,2)
mod_new:add(maxpool2)
mod_new:add(nn.SpatialMaxUnpooling(maxpool2))
mod_new:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
decode1 = nn.Tanh()
mod_new:add(decode1)
for i=0,2 do
	mod_new:add(mod_old:get(5+i))
end
decode2 = mod_new:get(13)
return mod_new


