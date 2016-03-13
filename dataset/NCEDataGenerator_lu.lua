
local NCEDataGenerator = torch.class('NCEDataGenerator')

function NCEDataGenerator:__init(vocab, nneg, power, normalizeUNK, tableSize)
  power = power or 1.0
  if normalizeUNK == nil then normalizeUNK = true end
  self.nneg = nneg
  self.tableSize = tableSize or 1e8
  
  self.unigramProbs, self.unigramBins = self:initUnigramProbs(vocab, self.tableSize, power, normalizeUNK)
  print('probs sum')
  print(self.unigramProbs:sum())
  print(self.unigramProbs:size(1))
  print('bins last')
  print(self.unigramBins[#self.unigramBins])
  print(#self.unigramBins)
end

function NCEDataGenerator:initUnigramProbs(vocab, tableSize, power, normalizeUNK)
  print('power', power)
  print('normalizeUNK', normalizeUNK)
  
  local freqs = vocab.freqs
  local uniqUNK = vocab.uniqUNK
  local unkID = vocab.UNK
  local word2idx = vocab.word2idx
  local vocabSize = vocab.nvocab
  
  if normalizeUNK then freqs[unkID] = math.ceil( freqs[unkID] / uniqUNK ) end
  
  local ifreqs = torch.LongTensor(freqs)
  local pfreqs = ifreqs:double():pow(power)
  
  local total = pfreqs:sum()
  local acc, i = pfreqs[1], 1
  local thres = acc / total
  local tableBins = {}
  local maxBinSize = 1e5
  assert(tableSize % maxBinSize == 0)
  local bins = torch.IntTensor(tableSize)
  local offset = 0
  for a = 1, tableSize do
    if a / tableSize > thres then
      i = i + 1
      if i > vocabSize then i = vocabSize end
      acc = acc + pfreqs[i]
      thres = acc / total
    end
    tableBins[a - offset] = i
    if a % maxBinSize == 0 then
      bins[{ {offset + 1, a} }] = torch.IntTensor(tableBins)
      offset = offset + maxBinSize
    end
  end
  
  local uprobs = pfreqs:div( pfreqs:sum() )
  while uprobs:sum() ~= 1 do
    uprobs = pfreqs:div( pfreqs:sum() )
  end
  
  return uprobs, bins
end

function NCEDataGenerator:getYNegProbs(y, useGPU)
  local probs = self.unigramProbs
  local bins = self.unigramBins
  local nneg = self.nneg
  
  assert(y:dim() == 2)
  local rnds = (torch.DoubleTensor(y:size(1) * y:size(2) * nneg):uniform(0, 1) * self.tableSize + 1):long()
  local y_neg = bins:index(1, rnds):long()
  
  local y_ = y:reshape(y:size(1) * y:size(2))
  y_[y_:eq(0)] = 1
  local y_prob = probs:index(1, y_):reshape(y:size(1), y:size(2))
  local y_neg_prob = probs:index(1, y_neg):reshape(y:size(1), y:size(2), nneg)
  y_neg = y_neg:reshape(y:size(1), y:size(2), nneg)
  
  if useGPU then
    return y_neg:cuda(), y_prob:cuda(), y_neg_prob:cuda()
  else
    return y_neg, y_prob:float(), y_neg_prob:float()
  end
end
