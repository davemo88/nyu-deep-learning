require 'nn'
require 'image'
require 'optim'
require 'cunn'
require('batchflip.lua')


mod_old = torch.load('trained/swwae_simple40.t7b'):get(3)
mod_new = nn.sequential()
for (i = 1,4) do
	mod_new:add(mod_old:get(i))
end	
mod_new:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
mod_new:add(nn.Batchnormalization())
mod_new:add(nn.Tanh())
mod_new:add(Dropout(0.4))
maxpool2 = nn.SpatialMaxPooling(2,2,2,2)
mod_new:add(maxpool2)
unpool2 = nn.SpatialMaxUnpooling(maxpool2)
mod_new:add(maxpool2)
mod_new:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
mod_new:add(nn.Tanh())
for (i=1,3) do
	mod_new:add(mod_old:get(5+i))
end

return mod_new


