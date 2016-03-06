require 'xlua'
require 'optim'
require 'cunn'
require 'batchflip.lua'

aug = nn.Sequential()
aug:add(nn.Batchflip())
return aug