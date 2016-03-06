require 'nn'
require 'image'
require 'optim'
require 'cunn'
require('batchflip.lua')
dofile('unsupervised_model_simple.lua')


file = 'trained/swwae_simple40.t7b'
num_old = 1
new_layer = nn.Sequential()
pooling, encoder = Encoder(64,64)
new_layer:add(encoder)
new_layer:add(Decoder(pooling, 64,64))
mod = torch.load(file)
current = mod
for (i = 1,num_old-1) do
	current = current:get(2)
end	
current:insert(2,new_layer)

return mod


