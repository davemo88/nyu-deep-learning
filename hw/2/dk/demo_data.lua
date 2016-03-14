require 'cunn'

local data_verbose = false

function getdata(clump, inputsize, std)
   local clump = torch.load('/home/ubuntu/2/clumps/' .. clump .. '.t7b')
   local data = clump.data:double()--:cuda()
   -- local data = torch.DiskFile(datafile,'r'):binary():readObject()
   local dataset = {}

   setmetatable(dataset, {__index = function(self, index) 
         return {data[index], data[index], data[index]} 
      end})

   return dataset
end

--    local std = std or 0.2
--    local nsamples = data:size(1)
--    local channels = data:size(2)
--    local nrows = data:size(3)
--    local ncols = data:size(4)

-- --   print(nsamples, channels, nrows, ncols)

--    function dataset:size()
--       return nsamples
--    end

--    function dataset:selectPatch(nr,nc)
--       local imageok = false
--       if simdata_verbose then
-- 	 print('selectPatch')
--       end
--       while not imageok do
-- 	 --image index
-- 	 local i = math.ceil(torch.uniform(1e-12,nsamples))
-- 	 local im = data:select(1,i)
--     print(i, im:size())
-- 	 -- select some patch for original that contains original + pos
-- 	 local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
-- 	 local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
--     print(ri, ci)
-- 	 local patch = im:narrow(1,ri,nr)
-- 	 patch = patch:narrow(2,ci,nc)
-- 	 local patchstd = patch:std()
-- 	 if data_verbose then
-- 	    print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
-- 	 end
-- 	 if patchstd > std then
-- 	    if data_verbose then
-- 	       print(patch:min(),patch:max())
-- 	    end
-- 	    return patch,i,im
-- 	 end
--       end
--    end

--    local dsample = torch.Tensor(inputsize*inputsize)

--    function dataset:conv()
--       dsample = torch.Tensor(1,inputsize,inputsize)
--    end

--    setmetatable(dataset, {__index = function(self, index)
-- 				       local sample,i,im = self:selectPatch(inputsize, inputsize)
-- 				       dsample:copy(sample)
-- 				       return {dsample,dsample,im}
-- 				    end})
--    return dataset
-- end

-- dataset, dataset=createDataset(....)
-- nsamples, how many samples to display from dataset
-- nrow, number of samples per row for displaying samples
-- scale, scale at which to draw dataset
function displayData(dataset, nsamples, nrow, scale)
   require 'image'
   local nsamples = nsamples or 100
   local scale = scale or 1
   local nrow = nrow or 10

   local win = nil

   cntr = 1
   local ex = {}
   for i=1,nsamples do
      local exx = dataset[1]
      ex[cntr] = exx[1]:clone():unfold(1,math.sqrt(exx[1]:size(1)),math.sqrt(exx[1]:size(1)))
      cntr = cntr + 1
   end
   --return ex
   win = image.display{image=ex, padding=1, symmetric=true, scale=scale, win=win, nrow=nrow}
   return win
end
