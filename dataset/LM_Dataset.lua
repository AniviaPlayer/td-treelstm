
require 'torch'
require 'hdf5'
require 'shortcut'

local LM_Dataset = torch.class('LM_Dataset')

function LM_Dataset:__init(datasetPath)
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
  -- print(table.keys(self.vocab))
  xprintln('vocab size %d', self.vocab.nvocab)
  self.eos = self.vocab.word2idx['###eos###']
  xprintln('EOS id %d', self.eos)
end

function LM_Dataset:getVocabSize()
  return self.vocab.nvocab
end

function LM_Dataset:getTrainSize()
  return self.trainSize
end

function LM_Dataset:getValidSize()
  return self.validSize
end

function LM_Dataset:getTestSize()
  return self.testSize
end

function LM_Dataset:toBatch(sents, eos, bs)
  local maxn = 0
  for _, sent in ipairs(sents) do
    if sent:size(1) > maxn then
      maxn = sent:size(1)
    end
  end
  maxn = maxn + 1
  local nsent = #sents
  -- for x, in default x contains EOS tokens
  local x = torch.ones(maxn, bs):type('torch.IntTensor')
  -- local x = torch.ones(maxn, batchSize)
  x:mul(eos)
  local y = torch.zeros(maxn, bs):type('torch.IntTensor')
  -- local y = torch.zeros(maxn, batchSize)
  for i = 1, nsent do
    local senlen = sents[i]:size(1)
    x[{ {2, senlen + 1}, i }] = sents[i]
    y[{ {1, senlen}, i }] = sents[i]
    y[{ senlen + 1, i }] = eos
  end
  
  return x, y
end

function LM_Dataset:createBatch(label, batchSize)
  local h5in = self.h5in
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local N = index:dataspaceSize()[1]
  local eos = self.eos
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local sents = {}
      for i = istart, iend do
        local idx = index:partial({i, i}, {1, 2})
        local start, len = idx[1][1], idx[1][2]
        local sent = x_data:partial({start, start + len - 1})
        table.insert(sents, sent)
      end
      
      istart = iend + 1
      
      return self:toBatch(sents, eos, batchSize)
    end
  end
end


function LM_Dataset:close()
  self.h5in:close()
end
