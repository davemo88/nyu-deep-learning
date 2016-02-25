require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling
local MaxUnpooling = nn.SpatialMaxUnpooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)
maxpool1 = MaxPooling(2,2,2,2):ceil()
vgg:add(maxpool1)

ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
maxpool2 = MaxPooling(2,2,2,2):ceil()
vgg:add(maxpool2)

ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
maxpool3 = MaxPooling(2,2,2,2):ceil()
vgg:add(maxpool3)

ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
maxpool4 = MaxPooling(2,2,2,2):ceil()
vgg:add(maxpool4)

ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
maxpool5 = MaxPooling(2,2,2,2):ceil()
vgg:add(maxpool5)

-- maybe something goes here because they have a intermediate reconstruction loss 
-- in the what-where autoencoders paper
-- could be this: vgg:add(nn.View(512*3*3))

vgg:add(MaxUnpooling(maxpool5))
ConvBNReLU(512,512)
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))

vgg:add(MaxUnpooling(maxpool4))
ConvBNReLU(512,512)
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,256):add(nn.Dropout(0.4))

vgg:add(MaxUnpooling(maxpool3))
ConvBNReLU(256,256)
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,128):add(nn.Dropout(0.4))

vgg:add(MaxUnpooling(maxpool2))
ConvBNReLU(128,128)
ConvBNReLU(128,64):add(nn.Dropout(0.4))

vgg:add(MaxUnpooling(maxpool1))
ConvBNReLU(64,64)
ConvBNReLU(64,3):add(nn.Dropout(0.4))

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

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
