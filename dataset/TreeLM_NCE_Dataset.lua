
require 'torch'
require 'hdf5'
require 'shortcut'
require 'NCEDataGenerator'

local TreeLM_NCE_Dataset = torch.class('TreeLM_NCE_Dataset')

function TreeLM_NCE_Dataset:__init(datasetPath, nneg, power, normalizeUNK)
  self.vocab = torch.load(datasetPath:sub(1, -3) .. 'vocab.t7')
  xprintln('load vocab done!')
  self.h5in = hdf5.open(datasetPath, 'r')
  
  local function getLength(label)
    local index = self.h5in:read(string.format('/%s/index', label))
    return index:dataspaceSize()[1]
  end
  self.trainSize = getLength('train')
  self.validSize = getLength('valid')
  self.testSize = getLength('test')
  xprintln('train size %d, valid size %d, test size %d', self.trainSize, self.validSize, self.testSize)
  xprintln('vocab size %d', self.vocab.nvocab)
  self.UNK = self.vocab.UNK
  xprintln('unknown word token is %d', self.UNK)
  
  self.ncedata = NCEDataGenerator(self.vocab, nneg, power, normalizeUNK)
end

function TreeLM_NCE_Dataset:getVocabSize()
  return self.vocab.nvocab
end

function TreeLM_NCE_Dataset:getTrainSize()
  return self.trainSize
end

function TreeLM_NCE_Dataset:getValidSize()
  return self.validSize
end

function TreeLM_NCE_Dataset:getTestSize()
  return self.testSize
end

function TreeLM_NCE_Dataset:toBatch(xs, ys, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 4):type(dtype)
  x:mul(self.UNK)
  x[{ {}, {}, 4 }] = torch.linspace(2, maxn + 1, maxn):resize(maxn, 1):expand(maxn, batchSize)
  local nsent = #ys
  local y = torch.zeros(maxn, batchSize):type(dtype)
  for i = 1, nsent do
    local sx, sy = xs[i], ys[i]
    x[{ {1, sx:size(1)}, i, {} }] = sx
    y[{ {1, sy:size(1)}, i }] = sy
  end
  
  return x, y
end

function TreeLM_NCE_Dataset:createBatch(label, batchSize, useNCE)
  local h5in = self.h5in
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local y_data = h5in:read(string.format('/%s/y_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local N = index:dataspaceSize()[1]
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local xs = {}
      local ys = {}
      for i = istart, iend do
        local idx = index:partial({i, i}, {1, 2})
        local start, len = idx[1][1], idx[1][2]
        local x = x_data:partial({start, start + len - 1}, {1, 4})
        local y = y_data:partial({start, start + len - 1})
        table.insert(xs, x)
        table.insert(ys, y)
      end
      
      istart = iend + 1
      
      local x, y = self:toBatch(xs, ys, batchSize)
      if useNCE then
        local mask = y:ne(0):float()
        y[y:eq(0)] = 1
        local y_neg, y_prob, y_neg_prob = self.ncedata:getYNegProbs(y)
        return x, y, y_neg, y_prob, y_neg_prob, mask
      else
        return x, y
      end
    end
  end
end

function TreeLM_NCE_Dataset:close()
  self.h5in:close()
end
