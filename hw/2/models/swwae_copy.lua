require 'nn'

local model = nn.Sequential()

-- building block
function Encoder(nInputPlane, nOutputPlane, nConv, pool)
  encoder = nn.Sequential()
  conv = nn.Sequential()
  conv:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  conv:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  for i=1,nConv-1 do
    conv:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    conv:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  end
  encoder:add(conv)
  pooling = nil
  if pool ~= 0 then
    pooling = nn.SpatialMaxPooling(pool,pool,pool,pool)
    encoder:add(pooling)
  end
  encoder:add(nn.ReLU())
  --encoder:add(nn.Dropout(0.5))
  return pooling, encoder
end

function Decoder(nInputPlane, nOutputPlane, pooling, nConv)
  decoder = nn.Sequential()
  if (pooling ~= nil) then
    decoder:add(nn.SpatialMaxUnpooling(pooling))
  end
  conv = nn.Sequential()
  for i=1,nConv-1 do
    conv:add(nn.SpatialConvolution(nInputPlane, nInputPlane, 3,3, 1,1, 1,1))
    conv:add(nn.SpatialBatchNormalization(nInputPlane, 1e-3))
  end
  conv:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  conv:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
  decoder:add(conv)
  decoder:add(nn.ReLU())
  return decoder
end




poolingA, encoderA =  Encoder(3,64,1,4)
model:add(encoder)
poolingB, encoderB = Encoder(64,64,1,3)
model:add(encoderB)
poolingC, encoderC = Encoder(64,128,2,2)
poolingD, encoderD = Encoder(128,256,3,0)
--poolingE, encoderE = Encoder(256,512,3,2)
cae_A = nn.Sequential()
cae_A:add(encoderA)
cae_B = nn.Sequential()
cae_A:add(cae_B)
cae_B:add(encoderB)
cae_C = nn.Sequential()
cae_B:add(cae_C)
cae_C:add(encoderC)
cae_D = nn.Sequential()
cae_C:add(cae_D)
cae_D:add(encoderD)
--cae_E = nn.Sequential()
--cae_D:add(cae_E)
--cae_E:add(encoderE)
--cae_E:add(Decoder(512,256,poolingE,3))
cae_D:add(Decoder(256,128,poolingD,3))
cae_C:add(Decoder(128,64,poolingC,2))
cae_B:add(Decoder(64,64,poolingB,1))
dec_D = Decoder(64,3,poolingA,1)
dec_D:remove(dec_D:size())
cae_A:add(dec_D)


model = cae_A

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
