
include '../utils/shortcut.lua'

require 'hdf5'

local LM_Dataset = {}

function LM_Dataset.toBatch(sents, eos)
  local maxn = 0
  for _, sent in ipairs(sents) do
    if sent:size(1) > maxn then
      maxn = sent:size(1)
    end
  end
  maxn = maxn + 1
  local batchSize = #sents
  -- for x, in default x contains EOS tokens
  local x = torch.ones(maxn, batchSize):type('torch.IntTensor')
  x:mul(eos)
  local y = torch.zeros(maxn, batchSize):type('torch.IntTensor')
  for i = 1, batchSize do
    local senlen = sents[i]:size(1)
    x[{ {2, senlen + 1}, i }] = sents[i]
    y[{ {1, senlen}, i }] = sents[i]
    y[{ senlen + 1, i }] = eos
  end
  
  return x, y
end

function LM_Dataset.createBatch(h5in, label, batchSize, eos)
  -- local h5in = hdf5.open(h5InFile, 'r')
  local x_data = h5in:read(string.format('/%s/x_data', label))
  local index = h5in:read(string.format('/%s/index', label))
  local N = index:dataspaceSize()[1]
  
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
      
      return LM_Dataset.toBatch(sents, eos)
    else
      h5in:close()
    end
  end
end

return LM_Dataset
