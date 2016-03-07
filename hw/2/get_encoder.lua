require 'nn'
require 'image'
require 'optim'
require 'cunn'
--dofile './batchflip.lua'
num_layers = 2

file = "trained/cae_simple10.t7b"
--augmentation = dofile('augmentation.lua')

mod_old = torch.load(file)

print(mod_old)

mod_new = nn.Sequential()
--mod_new:add(augmentation)

print(mod_old:get(1))

mod_new:add(mod_old:get(1))

print(mod_new:get(1):get(5).output:size())

s = mod_new:get(1):get(5).output:size()

linear_s = s[2]*s[3]*s[4]

mod_new:add(nn.View(linear_s))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(linear_s,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
mod_new:add(classifier)

print(mod_new)

return mod_new