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
  local batch_size = 4000
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

  self.batches = {}

  records = raw_extra.data[1]

  num_batches = extra_size / batch_size

  for i=1, num_batches do

    print('batch_num: ' .. i)

    self.batches[i] = {
      data = torch.ByteTensor(batch_size, channel, height, width),
      size = function() return batch_size end
    }

    start_idx = (i-1) * batch_size + 1
    end_idx = start_idx + batch_size - 1

    --print(start_idx, end_idx)

    for j = start_idx, end_idx do

      batch_idx = (j % 4000) + 1

      self.batches[i].data[batch_idx]:copy(records[j])
    end

    self.batches[i].data = self.batches[i].data:float()
    collectgarbage()

  end

-- get the labled examples as their own batch
  train_batch = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return train_size end
  }
  train_batch.data, train_batch.labels = parseDataLabel(raw_train.data,
                                                      train_size, channel, height, width)

-- convert from ByteTensor to Float
  train_batch.data = train_batch.data:float()
  collectgarbage()

  self.batches[num_batches+1] = train_batch

end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  local batches = self.batches

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  local mean_u = 0
  local mean_v = 0
  local var_u = 0
  local var_v = 0
  local mean_var_u = 0
  local mean_var_v = 0

  local num_batches = table.getn(batches)

  for i=1, num_batches do
    b = batches[i]
    print('==> batch ' .. i)
    -- preprocess batches
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
    print('==> get batch u stats for global normalization')
    b.mean_u = b.data:select(2,2):mean()
    b.var_u = b.data:select(2,2):var()
    mean_u = mean_u + b.mean_u
    var_u = var_u + b.var_u

    print('==> get batch v stats for global normalization')
    b.mean_v = b.data:select(2,3):mean()
    b.var_v = b.data:select(2,3):var()
    mean_v = mean_v + b.mean_v
    var_v = var_v + b.var_v
    collectgarbage()
  end

  print('==> compute global mean and var for u')
  mean_u = mean_u / num_batches
  for i=1, num_batches do
    mean_var_u = mean_var_u + (batches[i].mean_u - mean_u)^2
  end
  var_u = (var_u + mean_var_u) / num_batches

  print('==> compute global mean and var for v')
  mean_v = mean_v / num_batches
  for i=1, num_batches do
    mean_var_v = mean_var_v + (batches[i].mean_v - mean_v)^2
  end
  var_v = (var_v + mean_var_v) / num_batches

  print('==> normalize batches globally')
  for i=1, num_batches do
    b = batches[i]
    print('==> batch ' .. i)
    print('==> normalize u globally')
    b.data:select(2,2):add(-mean_u)
    b.data:select(2,2):div(math.sqrt(var_u))
    print('==> normalize v globally')
    b.data:select(2,3):add(-mean_v)
    b.data:select(2,3):div(math.sqrt(var_v))
  end

end

function Provider:save_batches()

  local batches = self.batches

  local num_batches = table.getn(batches)

  for i=1, num_batches do
    print('==> saving batch ' .. i)
    torch.save('batches/' .. i .. '.t7b', batches[i])
  end
end
