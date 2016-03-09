require 'nn'


local channels = 3
local height = 96
local width = 96
local input_dim = channels * height * width

output_dim = input_dim

-- encoder
local encoder = nn.Sequential()
encoder:add(nn.Linear(input_dim,output_dim))
encoder:add(nn.Tanh())

-- decoder
local decoder = nn.Sequential()
decoder:add(nn.Linear(output_dim,input_dim))

model = nn.Sequential()
model:add(encoder)
model:add(decoder)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(model)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return model
