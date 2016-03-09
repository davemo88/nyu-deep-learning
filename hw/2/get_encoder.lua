require 'nn'
require 'image'
require 'optim'
require 'cunn'
num_layers = 2

-- Load  CAE
file = "trained/cae_3L_6E.t7b"
mod_old = torch.load(file)



-- determine number of stacked autoencoders
depth  = 0
m = mod_old
while true do
  depth = depth + 1
  if (m:size() == 2) then
    break 
  end
  m = m:get(2)
end



mod_new = nn.Sequential()


-- Extract all encoder mdoules from old model
for i=1,depth do
	mod_new:add(mod_old:get(1))
	mod_old = mod_old:get(2)
end

print(mod_new)

-- Put a linear view on top of the last Encoder
s = mod_new:get(mod_new:size()).output:size()
linear_s = s[2]*s[3]*s[4]
mod_new:add(nn.View(linear_s))

-- Add a Classifier
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(linear_s,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
mod_new:add(classifier)
return mod_new