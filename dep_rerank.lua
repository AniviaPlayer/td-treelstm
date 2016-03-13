
require '.'
require 'shortcut'
require 'TreeLSTMNCELM'
require 'TreeLSTMLM'
require 'BiTreeLSTMLM'
require 'hdf5'
require 'deptreeutils'

local model_utils = require 'model_utils'

-- currently only support TreeLSTMLM and TreeLSTMNCE
-- support BiTreeLSTMLM as well
local Reranker = torch.class('TreeLSTMLMDepReranker')

function Reranker:__init(modelPath, useGPU)
  local optsPath = modelPath:sub(1, -4) .. '.state.t7'
  local opts = torch.load(optsPath)
  xprintln('load state from %s done!', optsPath)
  
  self.useGPU = useGPU
  opts.useGPU = useGPU
  print(opts)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  if opts.model == 'TreeLSTMNCE' then
    self.rnnlm = TreeLSTMNCELM(opts)
  elseif opts.model == 'TreeLSTM' then
    self.rnnlm = TreeLSTMLM(opts)
  elseif opts.model == 'BiTreeLSTM' then
    self.rnnlm = BiTreeLSTMLM(opts)
  else
    error('currently only support TreeLSTMNCELM, TreeLSTMLM and BiTreeLSTMLM')
  end
  
  xprintln( 'load model from %s', modelPath )
  self.rnnlm:load(modelPath)
  xprintln( 'load model from %s done!', modelPath )
end

function Reranker:rerankBidirectional(testFile, outFile, batchSize)
  self.rnnlm:disableDropout()
  
  local logp_sents = {}
  local dataIter = Reranker.createBatchBidirectional(testFile, batchSize)
  local cnt = 0
  for x, y, lc, lc_mask in dataIter do
    -- yPred | size: (seqlen*bs, nvocab)
    local _, yPred = self.rnnlm:validBatch(x, y, lc, lc_mask)
    if self.useGPU then y = y:cuda() end
    local mask = self.useGPU and y:ne(0):cuda() or y:ne(0):double()
    y[y:eq(0)] = 1
    local y_ = y:view(y:size(1) * y:size(2), 1)
    local logps = yPred:gather(2, y_)   -- shape: seqlen*bs, 1
    local logp_sents_ = logps:cmul(mask):view(y:size(1), y:size(2)):sum(1):squeeze()
    for i = 1, logp_sents_:size(1) do
      if mask[{ {}, i }]:sum() ~= 0 then
        logp_sents[#logp_sents + 1] = logp_sents_[i]
        cnt = cnt + 1
      end
    end
    
    -- cnt = cnt + y:size(2)
    if cnt % 100 == 0 then
      xprintln('cnt = %d', cnt)
    end
  end
  
  xprintln('Totally %d trees', #logp_sents)
  local fout = io.open(outFile, 'w')
  for _, logp in ipairs(logp_sents) do
    fout:write( string.format('%f\n', logp) )
  end
  
  fout:close()
  
  self.rnnlm:enableDropout()
end


function Reranker:rerank(testFile, outFile, batchSize)
  self.rnnlm:disableDropout()
  
  local logp_sents = {}
  local dataIter = Reranker.createBatch(testFile, batchSize)
  local cnt = 0
  for x, y in dataIter do
    -- yPred | size: (seqlen*bs, nvocab)
    local _, yPred = self.rnnlm:validBatch(x, y)
    if self.useGPU then y = y:cuda() end
    local mask = self.useGPU and y:ne(0):cuda() or y:ne(0):double()
    y[y:eq(0)] = 1
    local y_ = y:view(y:size(1) * y:size(2), 1)
    local logps = yPred:gather(2, y_)   -- shape: seqlen*bs, 1
    local logp_sents_ = logps:cmul(mask):view(y:size(1), y:size(2)):sum(1):squeeze()
    for i = 1, logp_sents_:size(1) do
      if mask[{ {}, i }]:sum() ~= 0 then
        logp_sents[#logp_sents + 1] = logp_sents_[i]
        cnt = cnt + 1
      end
    end
    
    -- cnt = cnt + y:size(2)
    if cnt % 100 == 0 then
      xprintln('cnt = %d', cnt)
    end
  end
  
  xprintln('Totally %d trees', #logp_sents)
  local fout = io.open(outFile, 'w')
  for _, logp in ipairs(logp_sents) do
    fout:write( string.format('%f\n', logp) )
  end
  
  fout:close()
  
  self.rnnlm:enableDropout()
end

function Reranker.toHDF5(vocabFile, testFile, bidirectional)
  print('load vocab ...')
  local vocab = torch.load(vocabFile)
  printf('load vocab done! %s\n', vocabFile)
  
  local testOutFile = testFile .. '.h5'
  local h5out = hdf5.open(testOutFile, 'w')
  if bidirectional then
    print('bidirectional Tree Model')
    DepTreeUtils.conllx2hdf5Bidirectional(testFile, h5out, 'test', vocab, 123456789)
  else
    DepTreeUtils.conllx2hdf5(testFile, h5out, 'test', vocab, 123456789)
  end
  
  printf('create testset done! %s\n', testOutFile)
  h5out:close()
end

function Reranker.toBatchBidirectional(xs, ys, lcs, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 5):type(dtype)
  -- x:mul(self.UNK)
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
    lchild:resize(maxLcSeqLen, lcBatchSize):fill(1)   -- UNK should be 1
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

function Reranker.toBatch(xs, ys, batchSize)
  local dtype = 'torch.LongTensor'
  local maxn = 0
  for _, y_ in ipairs(ys) do
    if y_:size(1) > maxn then
      maxn = y_:size(1)
    end
  end
  local x = torch.ones(maxn, batchSize, 4):type(dtype)
  -- x:mul(1)
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

function Reranker.createBatchBidirectional(testH5File, batchSize)
  local h5in = hdf5.open(testH5File, 'r')
  local label = 'test'
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
      
      return Reranker.toBatchBidirectional(xs, ys, lcs, batchSize)
    else
      h5in:close()
    end
  end
end

function Reranker.createBatch(testH5File, batchSize)
  local h5in = hdf5.open(testH5File, 'r')
  local label = 'test'
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
      
      local x, y = Reranker.toBatch(xs, ys, batchSize)
      
      return x, y
    else
      h5in:close()
    end
  end
end

function Reranker.rankAndEval(baseFile, baseScoreFile, scoreFile, goldFile, K, standard, searchK)
  local function conllxIterator(infile)
    local fin = io.open(infile)
    local bufs = {}
    
    return function()
      while true do
        local line = fin:read()
        if line == nil then
          fin:close()
          break
        end
        line = line:trim()
        if line:len() == 0 then
          local rlines = {}
          for i, buf in ipairs(bufs) do
            rlines[i] = buf
          end
          table.clear(bufs)
          
          return rlines
        else
          table.insert(bufs, line)
        end
      end
    end
    
  end
  
  -- show UAS and LAS scores for baseline system
  local baseIter = conllxIterator(baseFile)
  local bsin = io.open(baseScoreFile, 'r')
  local baseTopFile = baseFile .. '.top1'
  local bout = io.open(baseTopFile, 'w')
  while true do
    local line = bsin:read()
    if line == nil then break end
    local scores = line:splitc('\t ')
    for i, score in ipairs(scores) do
      local blines = baseIter()
      if i == 1 then
        for _, bline in ipairs(blines) do
          bout:write( string.format('%s\n', bline) )
        end
        bout:write('\n')
      end
    end
    
  end
  bsin:close()
  bout:close()
  
  xprintln('==Baseline Scores (%s)==', standard)
  local baseLAS, baseUAS = Reranker.getAttScore(baseTopFile, goldFile, standard)
  printf('LAS = %.2f, UAS = %.2f\n', baseLAS, baseUAS)
  
  local function rerankTopK(K)
    -- show UAS and LAS scores for re-ranked system
    local baseIter = conllxIterator(baseFile)
    local bsin = io.open(baseScoreFile, 'r')
    local sin = io.open(scoreFile, 'r')
    local rerankFile = string.format('%s.rerank.%d', baseFile, K) -- baseFile .. '.rerank'
    local rout = io.open(rerankFile, 'w')
    
    while true do
      local line = bsin:read()
      if line == nil then break end
      local bscores = line:splitc('\t ')
      local rerankScores = {}
      local bestScore = -1e309
      local bestScoreIndex = -1
      for i, bscore in ipairs(bscores) do
        local rankScore = tonumber(sin:read())
        rerankScores[#rerankScores + 1] = rankScore
        if i <= K then
          if rankScore > bestScore then
            bestScore = rankScore
            bestScoreIndex = i
          end
        end
      end
      
      for i, bscore in ipairs(bscores) do
        local blines = baseIter()
        if i == bestScoreIndex then
          for _, bline in ipairs(blines) do
            rout:write( string.format('%s\n', bline) )
          end
          rout:write('\n')
        end
      end
      
    end
    
    bsin:close()
    sin:close()
    rout:close()
    
    -- xprintln('==Rerank Scores==')
    -- os.execute(string.format('./conllx_scripts/eval_new.pl -s %s -g %s -q', rerankFile, goldFile))
    -- os.execute(string.format('./conllx_scripts/eval.lua --sysFile %s --goldFile %s', rerankFile, goldFile))
    local LAS, UAS = Reranker.getAttScore(rerankFile, goldFile, standard)
    
    return LAS, UAS
  end
  
  local bestUAS, bestLAS, bestK = 0, 0, 0
  if searchK then
    for k = 1, K do
      local rerankLAS, rerankUAS = rerankTopK(k)
      if rerankUAS >= bestUAS then
        bestK, bestUAS, bestLAS = k, rerankUAS, rerankLAS
      end
      printf('K = %d, LAS = %.2f, UAS = %.2f\n', k, rerankLAS, rerankUAS)
    end
  else
    bestK = K
    bestLAS, bestUAS = rerankTopK(K)
  end
  
  xprintln('\n\n==Baseline Scores (%s)==', standard)
  printf('LAS = %.2f, UAS = %.2f\n', baseLAS, baseUAS)
  xprintln('==Rerank Scores (%s)==', standard)
  printf('best K = %d, LAS = %.2f (+ %.2f), UAS = %.2f (+ %.2f)\n', bestK, bestLAS, (bestLAS - baseLAS), bestUAS, (bestUAS - baseUAS))
end

function Reranker.getAttScore(sysFile, goldFile, standard)
  assert(standard == 'conllx' or standard == 'stanford', 'only support conllx and stanford dependency')
  if standard == 'stanford' then
    local conllx_eval = require('conllx_eval')
    local _, _, LAS, UAS = conllx_eval.eval(sysFile, goldFile)
    LAS = tonumber(string.format('%.2f', LAS))
    UAS = tonumber(string.format('%.2f', UAS))
    
    return LAS, UAS
  else
    local function getNum(line)
      local s, _ = line:find('=')
      return tonumber(line:sub(s + 1, -2))
    end
    local LAS, UAS
    local cmd = string.format('./conllx_scripts/eval_new.pl -s %s -g %s -q', sysFile, goldFile)
    local file = io.popen(cmd)
    for line in file:lines() do
      line = line:trim()
      if line:find('Labeled   attachment score') then
        LAS = getNum(line)
      elseif line:find('Unlabeled attachment score') then
        UAS = getNum(line)
      end
    end
    
    return LAS, UAS
  end
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Reranking for Dependency Parsers ======')
  cmd:text()
  cmd:text('Options for scoring')
  cmd:option('--useGPU', false, 'do you want to run this on a GPU?')
  cmd:option('--modelPath', '', 'path for the trained model; modelPath.state.t7 should be the option of the model')
  cmd:option('--vocab', '', 'vocabulary file created from the training set')
  cmd:option('--baseFile', '', 'test file for reranking (CoNLL X format)')
  cmd:option('--scoreFile', '', 'dependency trees with ranking scores')
  cmd:option('--batchSize', 20, 'batch size')
  
  cmd:text()
  cmd:text('Options for rerank')
  cmd:option('--noRescore', false, 'will not rescore candidates')
  cmd:option('--k', 10, 'top k trees will be reranked')
  cmd:option('--baseScoreFile', '', 'score file for the baseline system')
  -- cmd:option('--scoreFile', '', 'score file of this model')
  cmd:option('--goldFile', '', 'gold standard conll x file')
  cmd:option('--standard', 'stanford', 'options: stanford, conllx')
  cmd:option('--searchk', false, 'search from 1 to K; Note that this option can only be used on devset!!!')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  print(opts)
  if not opts.noRescore then
    assert(opts.useGPU, 'currently only support GPU mode!')
    local reranker = TreeLSTMLMDepReranker(opts.modelPath, opts.useGPU)
    -- preprocessing
    TreeLSTMLMDepReranker.toHDF5(opts.vocab, opts.baseFile, reranker.rnnlm.name:starts('BiTree'))
    
    local h5testFile = opts.baseFile .. '.h5'
    if reranker.rnnlm.name:starts('BiTree') then
      reranker:rerankBidirectional(h5testFile, opts.scoreFile, opts.batchSize)
    else
      reranker:rerank(h5testFile, opts.scoreFile, opts.batchSize)
    end
  end
  
  TreeLSTMLMDepReranker.rankAndEval(opts.baseFile, opts.baseScoreFile, opts.scoreFile, 
    opts.goldFile, opts.k, opts.standard, opts.searchk)
end

main()

