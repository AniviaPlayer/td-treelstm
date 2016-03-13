
require 'torch'
require 'hdf5'
require 'shortcut'
require 'NCEDataGenerator'

local BidTreeLM_NCE_Dataset = torch.class('BidTreeLM_NCE_Dataset')

function BidTreeLM_NCE_Dataset:__init(datasetPath, nneg, power, normalizeUNK)
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

function BidTreeLM_NCE_Dataset:getVocabSize()
  return self.vocab.nvocab
end

function BidTreeLM_NCE_Dataset:getTrainSize()
  return self.trainSize
end

function BidTreeLM_NCE_Dataset:getValidSize()
  return self.validSize
end

function BidTreeLM_NCE_Dataset:getTestSize()
  return self.testSize
end

function BidTreeLM_NCE_Dataset:toBatch(xs, ys, lcs, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 5):type(dtype)
  x:mul(self.UNK)
  x[{ {}, {}, 4 }] = torch.linspace(2, maxn + 1, maxn):resize(maxn, 1):expand(maxn, batchSize)
  x[{ {}, {}, 5 }] = 0    -- in default, I don't want them to have 
  local nsent = #ys
  local y = torch.zeros(maxn, batchSize):type(dtype)
  for i = 1, nsent do
    local sx, sy = xs[i], ys[i]
    x[{ {1, sx:size(1)}, i, {} }] = sx
    y[{ {1, sy:size(1)}, i }] = sy
  end
  
  -- for left children
  assert(#lcs == #xs, 'should be the same!')
  local lcBatchSize = 0
  local maxLcSeqLen = 0
  for _, lc in ipairs(lcs) do
    if lc:dim() ~= 0 then
      lcBatchSize = lcBatchSize + 1
      maxLcSeqLen = math.max(maxLcSeqLen, lc:size(1))
    end
  end
  local lchild = torch.Tensor():type(dtype)
  local lc_mask = torch.FloatTensor()
  
  if lcBatchSize ~= 0 then
    lchild:resize(maxLcSeqLen, lcBatchSize):fill(self.UNK)
    lc_mask:resize(maxLcSeqLen, lcBatchSize):fill(0)
    local j = 0
    for i, lc in ipairs(lcs) do
      if lc:dim() ~= 0 then
        j = j + 1
        lchild[{ {1, lc:size(1)}, j }] = lc[{ {}, 1 }]
        lc_mask[{ {1, lc:size(1)}, j }] = lc[{ {}, 2 }] + 1
        local xcol = x[{ {}, i, 5 }]
        local idxs = xcol:ne(0)
        xcol[idxs] = (xcol[idxs] - 1) * lcBatchSize + j
      end
    end
  end
  
  return x, y, lchild, lc_mask
end

function BidTreeLM_NCE_Dataset:createBatch(label, batchSize, useNCE)
  local h5in = self.h5in
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local y_data = h5in:read(string.format('/%s/y_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local l_data = h5in:read( string.format('/%s/l_data', label) )
  local lindex = h5in:read( string.format('/%s/lindex', label) )
  local N = index:dataspaceSize()[1]
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local xs = {}
      local ys = {}
      local lcs = {}
      for i = istart, iend do
        local idx = index:partial({i, i}, {1, 2})
        local start, len = idx[1][1], idx[1][2]
        local x = x_data:partial({start, start + len - 1}, {1, 5})
        local y = y_data:partial({start, start + len - 1})
        table.insert(xs, x)
        table.insert(ys, y)
        
        local lidx = lindex:partial({i, i}, {1, 2})
        local lstart, llen = lidx[1][1], lidx[1][2]
        local lc
        if llen == 0 then
          lc = torch.IntTensor()  -- to be the same type as l_data
        else
          lc = l_data:partial({lstart, lstart + llen - 1}, {1, 2})
        end
        table.insert(lcs, lc)
      end
      
      istart = iend + 1
      
      local x, y, lchild, lc_mask = self:toBatch(xs, ys, lcs, batchSize)
      if useNCE then
        local mask = y:ne(0):float()
        y[y:eq(0)] = 1
        local y_neg, y_prob, y_neg_prob = self.ncedata:getYNegProbs(y)
        return x, y, lchild, lc_mask, y_neg, y_prob, y_neg_prob, mask
      else
        return x, y, lchild, lc_mask
      end
    end
  end
end

function BidTreeLM_NCE_Dataset:close()
  self.h5in:close()
end
