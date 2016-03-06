require 'nn'
require 'image'
require 'optim'
require 'cunn'
require('batchflip.lua')
num_layers = 2

file = "trained/swwae_simple40.t7b"
augmentation = dofile('augmentation.lua')

mod_old = torch.load(file)

mod_new = nn.Sequential()
mod_new:add(augmentation)

for i=1,num_layers-1 do
    mod_new:add(mod_old:get(1))
    mod_old = mod_old:get(2)
end

dim = mod_old:get(1):get(4):output:size()
mod_new:add(mod_old:get(1))

mod_new:add(nn.Linear(dim, 10))

return mod_new

