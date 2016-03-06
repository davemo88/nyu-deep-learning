require 'nn'
require 'image'
require 'xlua'
require 'math'

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
  local train_size = 4000
  local extra_size = 100000
  local clump_size = 4000
  local channel = 3
  local height = 96
  local width = 96

  -- download dataset
  if not paths.dirp('stl-10') then
     os.execute('mkdir stl-10')
     local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
     }

     os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end

  local raw_train = torch.load('stl-10/train.t7b')
  local raw_extra = torch.load('stl-10/extra.t7b')

  self.clumps = {}

  records = raw_extra.data[1]

  num_clumps = extra_size / clump_size

  for i=1, num_clumps do

    print('clump_num: ' .. i)

    self.clumps[i] = {
      data = torch.ByteTensor(clump_size, channel, height, width),
      size = function() return clump_size end
    }

    start_idx = (i-1) * clump_size + 1
    end_idx = start_idx + clump_size - 1

    --print(start_idx, end_idx)

    for j = start_idx, end_idx do

      clump_idx = (j % 4000) + 1

      self.clumps[i].data[clump_idx]:copy(records[j])
    end

    self.clumps[i].data = self.clumps[i].data:float()
    collectgarbage()

  end

-- get the labled examples as their own clump
  train_clump = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return train_size end
  }
  train_clump.data, train_clump.labels = parseDataLabel(raw_train.data,
                                                      train_size, channel, height, width)

-- convert from ByteTensor to Float
  train_clump.data = train_clump.data:float()
  collectgarbage()

  self.clumps[num_clumps+1] = train_clump

end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  local clumps = self.clumps

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  local mean_u = 0
  local mean_v = 0
  local var_u = 0
  local var_v = 0
  local mean_var_u = 0
  local mean_var_v = 0

  local num_clumps = table.getn(clumps)

  for i=1, num_clumps do
    b = clumps[i]
    print('==> clump ' .. i)
    -- preprocess clumps
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    print('==> normalize y locally:')
    for i = 1,b:size() do
       xlua.progress(i, b:size())
       -- rgb -> yuv
       local rgb = b.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[1] = normalization(yuv[{{1}}])
       b.data[i] = yuv
       collectgarbage()
    end
    -- normalize u globally:
    print('==> get clump u stats for global normalization')
    b.mean_u = b.data:select(2,2):mean()
    b.var_u = b.data:select(2,2):var()
    mean_u = mean_u + b.mean_u
    var_u = var_u + b.var_u

    print('==> get clump v stats for global normalization')
    b.mean_v = b.data:select(2,3):mean()
    b.var_v = b.data:select(2,3):var()
    mean_v = mean_v + b.mean_v
    var_v = var_v + b.var_v
    collectgarbage()
  end

  print('==> compute global mean and var for u')
  mean_u = mean_u / num_clumps
  for i=1, num_clumps do
    mean_var_u = mean_var_u + (clumps[i].mean_u - mean_u)^2
  end
  var_u = (var_u + mean_var_u) / num_clumps

  print('==> compute global mean and var for v')
  mean_v = mean_v / num_clumps
  for i=1, num_clumps do
    mean_var_v = mean_var_v + (clumps[i].mean_v - mean_v)^2
  end
  var_v = (var_v + mean_var_v) / num_clumps

  print('==> normalize clumps globally')
  for i=1, num_clumps do
    b = clumps[i]
    print('==> clump ' .. i)
    print('==> normalize u globally')
    b.data:select(2,2):add(-mean_u)
    b.data:select(2,2):div(math.sqrt(var_u))
    print('==> normalize v globally')
    b.data:select(2,3):add(-mean_v)
    b.data:select(2,3):div(math.sqrt(var_v))
  end

end

function Provider:save_clumps()

  local clumps = self.clumps

  local num_clumps = table.getn(clumps)

  for i=1, num_clumps do
    print('==> saving clump ' .. i)
    torch.save('clumps/' .. i .. '.t7b', clumps[i])
  end
end
