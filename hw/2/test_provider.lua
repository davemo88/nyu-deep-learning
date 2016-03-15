
require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local nclumps = 10
  local trsize = 800
  local channel = 3
  local height = 96
  local width = 96


  local raw_test = torch.load('test.t7b')

  -- load and parse dataset
  self.testData = {
     data = torch.Tensor(nclumps * trsize, channel, height, width),
     size = function() return nclumps * trsize end
  }
  for i=1, nclumps do
    for j=1, trsize do
      idx = (trsize * (i-1)) + j
      self.testData.data[idx]:copy(raw_test.data[i][j])
    end
  end

  -- local testData = self.testData

  -- convert from ByteTensor to Float
  self.testData.data = self.testData.data:float()
  -- self.testData.labels = self.testData.labels:float()
  collectgarbage()
end

function Provider:normalize(stats)
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess test set
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,testData:size() do
     xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-stats.mu)
  testData.data:select(2,2):div(stats.su)
  -- normalize v globally:
  testData.data:select(2,3):add(-stats.mv)
  testData.data:select(2,3):div(stats.sv)

  testData.mean_u = mean_u
  testData.std_u = std_u
  testData.mean_v = mean_v
  testData.std_v = std_v
end
