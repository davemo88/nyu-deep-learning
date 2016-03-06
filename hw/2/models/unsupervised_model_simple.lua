require 'nn'

local model = nn.Sequential()

-- building block
local function Encoder(nInputPlane, nOutputPlane)
  encoder = nn.Sequential()
  encoder:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  encoder:add(nn.ReLU())
  pooling = nn.SpatialMaxPooling(2,2,2,2)
  encoder:add(pooling)
  encoder:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  encoder:add(nn.Dropout(0.5))
  return pooling, encoder
end

local function Decoder(nInputPlane, nOutputPlane, pooling)
  decoder = nn.Sequential()
  decoder:add(nn.SpatialMaxUnpooling(pooling))
  decoder:add(nn.SpatialConvolution(nInputPlane,nOutputPlane, 3, 3, 1, 1, 1, 1))
  decoder:add(nn.ReLU())
  decoder:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  return decoder
end



pooling, encoder =  Encoder(3,64)
model:add(encoder)
decoder = Decoder(64,3, pooling)
decoder:delete(4)
decoder:delete(3)
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
