
require '.'
require 'shortcut'

require 'TreeLSTMLM'
require 'TreeLM_Dataset'

require 'TreeLSTMNCELM'
require 'TreeLM_NCE_Dataset'

require 'BiTreeLSTMLM'
require 'BidTreeLM_Dataset'

require 'BiTreeLSTMNCELM'
require 'BidTreeLM_NCE_Dataset'

require 'MLP'
require 'xqueue'

local Sampler = torch.class('TreeSampler')

function Sampler:__init(modelPath, useGPU)
  local optsPath = modelPath:sub(1, -4) .. '.state.t7'
  local opts = torch.load(optsPath)
  self.opts = opts
  xprintln('load state from %s done!', optsPath)
  
  self.useGPU = useGPU
  opts.useGPU = useGPU
  -- opts.seqLen = 10
  print(opts)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  if opts.model == 'TreeLSTMNCE' then
    self.treelstm = TreeLSTMNCELM(opts)
  elseif opts.model == 'BiTreeLSTMNCE' then
    self.treelstm = BiTreeLSTMNCELM(opts)
  elseif opts.model == 'TreeLSTM' then
    self.treelstm = TreeLSTMLM(opts)
  elseif opts.model == 'BiTreeLSTM' then
    self.treelstm = BiTreeLSTMLM(opts)
  else
    error('currently support TreeLSTMNCELM, BiTreeLSTMNCELM, TreeLSTMLM and BiTreeLSTMLM')
  end
  
  xprintln( 'load model from %s', modelPath )
  self.treelstm:load(modelPath)
  xprintln( 'load model from %s done!', modelPath )
  
  self:getOEmb()
end

function Sampler:getOEmb()
  local model_utils = require 'model_utils'
  self.treelstm.oemb = model_utils.get_linear(self.treelstm.softmax)
end

function Sampler:loadVocab(path)
  -- datasetPath:sub(1, -3) .. 'vocab.t7'
  path = path or self.opts.dataset:sub(1, -3) .. 'vocab.t7'
  self.vocab = torch.load(path)
  printf('load vocab from %s done! vocab size %d\n', path, self.vocab.nvocab)
end

function Sampler:generateSplit(label, h5out)
  local gxdata = string.format('/%s/x', label)
  local gydata = string.format('/%s/y', label)
  local gedata = string.format('/%s/e', label)
  local goedata = string.format('/%s/oe', label)
  local xOpt = hdf5.DataSetOptions()
  xOpt:setChunked(1024*10, self.opts.nhid)
  xOpt:setDeflate()
  local yOpt = hdf5.DataSetOptions()
  yOpt:setChunked(1024*10, 4)
  yOpt:setDeflate()
  local eOpt = hdf5.DataSetOptions()
  eOpt:setChunked(1024*10, self.opts.nin)
  eOpt:setDeflate()
  local oeOpt = hdf5.DataSetOptions()
  oeOpt:setChunked(1024*10, self.opts.nhid)
  oeOpt:setDeflate()
  
  local dataIter = self.lmdata:createBatch(label, self.opts.batchSize)
  local i_act, i_prev_h, i_cur_h = 2, 3, 4
  local isFirst = true
  local cnt = 0
  for x, y, lc, lc_mask in dataIter do
    local hids
    if lc then
      hids = self.treelstm:getHiddenStates(x, lc, lc_mask)
    else
      hids = self.treelstm:getHiddenStates(x)
    end
    local hs = hids:view(x:size(1) + 1, x:size(2), hids:size(2)):float()
    local xydata = {}
    local bs = x:size(2)
    local mask = y:ne(0):float()
    for j = 1, bs do
      local slen = mask[{ {}, j }]:sum()
      local acts = torch.zeros(slen+1, 4):int()
      local ows = torch.zeros(slen+1):int()
      local root_i = x[{ 1, j, i_prev_h }]
      assert(root_i == 1, 'it must be the first hidden states')
      ows[ root_i ] = x[{ 1, j, 1 }]
      for i = 1, slen do
        local act, prev_h = x[{ i, j, i_act }], x[{ i, j, i_prev_h }]
        acts[{ prev_h, act }] = 1
        local cur_h = x[{ i, j, i_cur_h }]
        ows[cur_h] = y[{ i, j }]
      end
      for i = 1, slen + 1 do
        local xdata = hs[{ i, j, {} }]
        local ydata = acts[i]
        local edata = self.treelstm.emb.weight[{ ows[i], {} }]:float()
        local oedata = self.treelstm.oemb.weight[{ ows[i], {} }]:float()
        table.insert(xydata, {xdata, edata, oedata, ydata})
      end
      --[[
      printf('j = %d, x = \n', j)
      print(x[{ {}, j, {} }])
      print('acts = ')
      print(acts)
      print('ows = ')
      print(ows)
      --]]
    end
    --[[
    print(mask:sum(1))
    print('xydata size = ')
    print(#xydata)
    break
    --]]
    local size = #xydata
    local xd = torch.zeros(size, self.opts.nhid):float()
    local ed = torch.zeros(size, self.opts.nin):float()
    local oed = torch.zeros(size, self.opts.nhid):float()
    local yd = torch.zeros(size, 4):int()
    for i = 1, size do
      -- xdata, edata, ydata
      xd[{ i, {} }] = xydata[i][1]
      ed[{ i, {} }] = xydata[i][2]
      oed[{ i, {} }] = xydata[i][3]
      yd[{ i, {} }] = xydata[i][4]
    end
    
    if isFirst then
      h5out:write(gxdata, xd, xOpt)
      h5out:write(gedata, ed, eOpt)
      h5out:write(goedata, oed, oeOpt)
      h5out:write(gydata, yd, yOpt)
      isFirst = false
    else
      h5out:append(gxdata, xd, xOpt)
      h5out:append(gedata, ed, eOpt)
      h5out:append(goedata, oed, oeOpt)
      h5out:append(gydata, yd, yOpt)
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then collectgarbage() end
    if cnt % 10 == 0 then
      printf('cnt = %d\n', cnt)
    end
  end
  
end

function Sampler:generateDataset(datasetIn, datasetOut)
  self.treelstm:disableDropout()
  local opts = self.opts
  local h5out = hdf5.open(datasetOut, 'w')
  if opts.model == 'TreeLSTM' then
    self.lmdata = TreeLM_Dataset(datasetIn)
  elseif opts.model == 'TreeLSTMNCE' then
    self.lmdata = TreeLM_NCE_Dataset(datasetIn, opts.nneg, opts.power, opts.normalizeUNK)
  elseif opts.model == 'BiTreeLSTM' then
    self.lmdata = BidTreeLM_Dataset(datasetIn)
  elseif opts.model == 'BiTreeLSTMNCE' then
    self.lmdata = BidTreeLM_NCE_Dataset(datasetIn, opts.nneg, opts.power, opts.normalizeUNK)
  end
  
  self:generateSplit('train', h5out)
  print('train split done!')
  self:generateSplit('valid', h5out)
  print('valid split done!')
  self:generateSplit('test', h5out)
  print('test split done!')
  
  self.treelstm:enableDropout()
  h5out:close()
end

local function generateDataset(model, datasetIn, datasetOut)
  local sampler = TreeSampler(model, true)
  sampler:generateDataset(datasetIn, datasetOut)
end

local Node = torch.class('TreeNode')
function Node:__init()
  self.leftChildren = {}
  self.rightChildren = {}
  self.parent = -1
  self.h = torch.Tensor()
  self.x = torch.Tensor()
  self.word = ''
  self.y_given_x = torch.Tensor()
  self.pos = 'NONE' -- can be 'left' or 'right'
end

function Sampler:loadClassifiers(classifierPath, useGPU)
  self.classifierOpts = {}
  self.classifiers = {}
  for i = 1, 4 do
    local modelPath = classifierPath:format(i)
    local optsPath = modelPath:sub(1, -4) .. '.state.t7'
    self.classifierOpts[i] = torch.load(optsPath)
    printf('classifier %d\n', i)
    self.classifierOpts[i].useGPU = useGPU
    print(self.classifierOpts[i])
    local classifier = MLP(self.classifierOpts[i])
    xprintln( 'load model from %s', modelPath )
    classifier:load(modelPath)
    xprintln( 'load model from %s done!', modelPath )
    self.classifiers[i] = classifier
  end
  
end

-- GEN_LEFT, GEN_RIGHT, GEN_LEFT_NEXT, GEN_RIGHT_NEXT = range(4)
Sampler.GEN_LEFT = 1
Sampler.GEN_RIGHT = 2
Sampler.GEN_NX_LEFT = 3
Sampler.GEN_NX_RIGHT = 4

function Sampler:gdata(x)
  if self.opts.useGPU then
    return x:cuda()
  else
    return x
  end
end

function Sampler:getRootNode()
  self.word2idx = self.vocab.word2idx
  self.idx2word = self.vocab.idx2word
  local rnode = TreeNode()
  rnode.word = '###root###'
  rnode.x = self.word2idx[rnode.word]
  local h0 = {}
  for i = 1, 2*self.opts.nlayers do
    h0[i] = self.treelstm.initStates[i][{ {1}, {} }]
  end
  rnode.h = h0
  
  return rnode
end

function Sampler:isGeneratable(act, h, x)
  local h_ = h[self.opts.nlayers*2]
  local e_x = self.treelstm.oemb.weight[{ {x}, {} }]
  local f = torch.cat(h_, e_x)
  local pred = self.classifiers[act]:predictBatch(f)
  
  return pred[{ 1, 2 }] > 0.5
end

function Sampler:sampleNext(y_probs)
  y_probs = y_probs:float()
  local y_probs_ = y_probs:view(-1):totable()
  local rnd = torch.uniform()
  local N = #y_probs_
  local wid = N
  for i = 1, N do
    rnd = rnd - y_probs_[i]
    if rnd <= 0 then
      wid = i
      break
    end
  end
  
  return wid, self.idx2word[wid]
end

function Sampler:getNextNode(act, u)
  local s_t, y_t = self.treelstm:fpropStep(act, u.x, u.h)
  local wid, word = self:sampleNext(y_t)
  local nd = TreeNode()
  nd.h = s_t
  nd.x = wid
  nd.word = word
  
  return nd
end

function Sampler:getNextNodeBi(act, u, lcs)
  local s_t, y_t = self.treelstm:fpropStep(act, u.x, u.h, lcs)
  local wid, word = self:sampleNext(y_t)
  local nd = TreeNode()
  nd.h = s_t
  nd.x = wid
  nd.word = word
  
  return nd
end

function Sampler:tree2sent(root, words)
  local nL = #root.leftChildren
  for i = nL, 1, -1 do
    self:tree2sent(root.leftChildren[i], words)
  end
  -- if word == '###unk###' then word = 'UNK' end
  local word = root.word
  local function normalizeWord(word)
    if word == '###unk###' then
      word = 'UNK'
    elseif word == '###root###' then
      word = 'ROOT'
    elseif word == '&' then
      word = '\\&'
    elseif word == '$' then
      word = '\\$'
    elseif word == '%' then
      word = '\\%'
    end
    
    return word
  end
  word = normalizeWord(word)
  table.insert(words, word)
  root.wsent_id = #words - 1
  local nR = #root.rightChildren
  for i = 1, nR do
    self:tree2sent(root.rightChildren[i], words)
  end
end

function Sampler:tree2string(root)
  local words = {}
  self:tree2sent(root, words)
  local s = ''
  for _, word in ipairs(words) do
    s = s .. string.format('%s ', word)
  end
  return s, words
end

function Sampler:tree2tlatex(root)
  local tlatex = ''
  local Q = XQueue()
  print(root)
  Q:push(root)
  local root_cnt_l, root_cnt_r = 0, 0
  while not Q:isEmpty() do
    local u = Q:pop()
    
    for _, c in ipairs(u.leftChildren) do
      if u.wsent_id == 0 then
        tlatex = tlatex .. string.format('\\deproot{%d}{ROOT}\n', c.wsent_id)
        root_cnt_l = root_cnt_l + 1
      else
        tlatex = tlatex .. string.format('\\depedge{%d}{%d}{}\n', u.wsent_id, c.wsent_id)
      end
      
      Q:push(c)
    end
    
    for _, c in ipairs(u.rightChildren) do
      if u.wsent_id == 0 then
        tlatex = tlatex .. string.format('\\deproot{%d}{ROOT}\n', c.wsent_id)
        root_cnt_r = root_cnt_r + 1
      else
        tlatex = tlatex .. string.format('\\depedge{%d}{%d}{}\n', u.wsent_id, c.wsent_id)
      end
      
      Q:push(c)
    end
    
  end
  
  assert(root_cnt_l < 1 and root_cnt_r == 1, 'only one ROOT allowed!')
  return tlatex
end

function Sampler:tree2latex(root, words)
  local latex = '\\begin{dependency}[theme = simple, arc angle=35]\n'
  latex = latex .. '\\begin{deptext}[column sep=0.03em]\n'
  local sent = ''
  for i, word in ipairs(words) do
    if i ~= 1 then
      sent = sent .. word
      local sep = i ~= #words and ' \\& ' or ' \\\\'
      sent = sent .. sep
    end
  end
  latex = latex .. sent .. '\n'
  latex = latex .. '\\end{deptext}\n'
  
  latex = latex .. self:tree2tlatex(root)
  
  latex = latex .. '\\end{dependency}\n'
  
  return latex
end

function Sampler:sampleTree(maxLen)
  maxLen = maxLen or 25
  local root = self:getRootNode()
  local Q = XQueue()
  Q:push(root)
  local abortByLength = false
  local cnt = 0
  
  while not Q:isEmpty() do
    local u = Q:pop()
    cnt = cnt + 1
    if cnt == maxlen then
      abortByLength = true
      break
    end
    
    printf('********u = %s, cnt = %d **********\n', u.word, cnt)
    
    -- generate left dependents
    local act = Sampler.GEN_LEFT
    if self:isGeneratable(act, u.h, u.x) then
      -- print('generate left')
      local nd1 = self:getNextNode(act, u)
      nd1.parent = u
      table.insert( u.leftChildren, nd1 )
      Q:push(nd1)
      
      act = Sampler.GEN_NX_LEFT
      local cur_nd = nd1
      while true do
        if not self:isGeneratable(act, cur_nd.h, cur_nd.x) then break end
        -- print('generate next left')
        local nx_nd = self:getNextNode(act, cur_nd)
        nx_nd.parent = u
        table.insert( u.leftChildren, nx_nd )
        Q:push(nx_nd)
        cur_nd = nx_nd
      end
    end
    
    -- generate right dependents
    act = Sampler.GEN_RIGHT
    if self:isGeneratable(act, u.h, u.x) then
      -- print('generate right')
      local nd1 = self:getNextNode(act, u)
      nd1.parent = u
      table.insert( u.rightChildren, nd1 )
      Q:push(nd1)
      
      act = Sampler.GEN_NX_RIGHT
      local cur_nd = nd1
      while true do
        if not self:isGeneratable(act, cur_nd.h, cur_nd.x) then break end
        -- print('generate next right')
        local nx_nd = self:getNextNode(act, cur_nd)
        nx_nd.parent = u
        table.insert( u.rightChildren, nx_nd )
        Q:push(nx_nd)
        cur_nd = nx_nd
      end
    end
    
  end
  
  local sent, words = self:tree2string(root)
  print(sent)
  print(words)
  local latex = self:tree2latex(root, words)
  print(latex)
  
  return sent, latex
end

function Sampler:sampleTreeBi(maxLen)
  maxLen = maxLen or 25
  local root = self:getRootNode()
  local Q = XQueue()
  Q:push(root)
  local abortByLength = false
  local cnt = 0
  
  while not Q:isEmpty() do
    local u = Q:pop()
    cnt = cnt + 1
    if cnt == maxlen then
      abortByLength = true
      break
    end
    
    printf('********u = %s, cnt = %d **********\n', u.word, cnt)
    
    local lcs = {}
    -- generate left dependents
    local act = Sampler.GEN_LEFT
    if self:isGeneratable(act, u.h, u.x) then
      -- print('generate left')
      local nd1 = self:getNextNode(act, u)
      nd1.parent = u
      table.insert( u.leftChildren, nd1 )
      Q:push(nd1)
      table.insert(lcs, nd1.x)
      
      act = Sampler.GEN_NX_LEFT
      local cur_nd = nd1
      while true do
        if not self:isGeneratable(act, cur_nd.h, cur_nd.x) then break end
        -- print('generate next left')
        local nx_nd = self:getNextNode(act, cur_nd)
        nx_nd.parent = u
        table.insert( u.leftChildren, nx_nd )
        Q:push(nx_nd)
        table.insert(lcs, nx_nd.x)
        cur_nd = nx_nd
      end
    end
    
    -- generate right dependents
    act = Sampler.GEN_RIGHT
    if self:isGeneratable(act, u.h, u.x) then
      -- print('generate right')
      local nd1 = self:getNextNodeBi(act, u, lcs)
      nd1.parent = u
      table.insert( u.rightChildren, nd1 )
      Q:push(nd1)
      
      act = Sampler.GEN_NX_RIGHT
      local cur_nd = nd1
      while true do
        if not self:isGeneratable(act, cur_nd.h, cur_nd.x) then break end
        -- print('generate next right')
        local nx_nd = self:getNextNode(act, cur_nd)
        nx_nd.parent = u
        table.insert( u.rightChildren, nx_nd )
        Q:push(nx_nd)
        cur_nd = nx_nd
      end
    end
    
  end
  
  local sent, words = self:tree2string(root)
  print(sent)
  print(words)
  local latex = self:tree2latex(root, words)
  print(latex)
  
  return sent, latex, #words
end

function Sampler:testClassifiers()
  -- self.treelstm.oemb.weight[{ ows[i], {} }]:float()
  local rootID = self.vocab.word2idx['###root###']
  local h_0 = self.treelstm.initStates[self.opts.nlayers * 2][{ 1, {} }]
  local e_r = self.treelstm.oemb.weight[{ rootID, {} }]
  local f0 = torch.cat(h_0, e_r)
  for i = 1, 4 do
    printf('i = %d\n', i)
    local pred = self.classifiers[i]:predictBatch(f0)
    print(pred:size())
    print(pred)
  end
  
  for i = 1, 4 do
    printf('i = %d\n', i)
    local pred = self.classifiers[i]:predictBatch(f0:view(1, -1))
    print(pred:size())
    print(pred)
  end
  
end

local function sampleTrees(modelPath, classifierPath, outputPath, seed, nsamples)
  local sampler = TreeSampler(modelPath, true)
  sampler:loadClassifiers(classifierPath, true)
  sampler:loadVocab()
  
  -- sampler:testClassifiers()
  
  seed = seed or 1
  torch.manualSeed(seed)
  if sampler.opts.useGPU then
    cutorch.manualSeed(seed)
  end
  
  local fout = io.open(outputPath, 'w')
  local dlatex = [[
\documentclass[11pt]{article}
\usepackage{xytree}
\usepackage{tikz-dependency}

\usepackage[top=1in, bottom=1in, left=0.005in, right=0.005in]{geometry}
\begin{document}
]]
  
  fout:write(dlatex)
  local nsent = 0
  for i = 1, nsamples do
    local sent, latex, nwords
    if sampler.opts.model:starts('BiTree') then
      sent, latex, nwords = sampler:sampleTreeBi(30)
    else
      sent, latex = sampler:sampleTree(30)
    end
    -- and sampler:sampleTreeBi(30) or sampler:sampleTree(30)
    local s = [[{\bf sentence %d \\}
]]
    if nwords <= 30 then
      nsent = nsent + 1
      fout:write(string.format(s, nsent))
      fout:write(sent .. '\\\\\n')
      fout:write(latex)
      fout:write('\n\n')
    end
    print('==============================')
    printf('===sample %d-th tree done!===\n', i)
    print('==============================')
    print('\n\n')
  end
  
  fout:write([[
\end{document}
]])
  fout:close()
end

local function main()
  --[[
  generateDataset('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/2layer_w_wo_we/model_1.0.w200.t7', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/dataset/penn_wsj.conllx.sort.h5', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/eot.penn_wsj.conllx.sort.h5')
  --]]
  
  --[[
  sampleTrees('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/2layer_w_wo_we/model_1.0.w200.t7',
    '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_classify/model.yt%d.x.oe.t7',
    's100.txt',
    1,
    100)
  --]]
  
  --[[
  generateDataset('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/bid_flush_best/model_1.0.200.lc200.t7', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/dataset/penn_wsj.conllx.bid.sort.h5', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/eot.penn_wsj.conllx.bid.sort.h5')
  --]]
  
  --[[
  sampleTrees('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/bid_flush_best/model_1.0.200.lc200.t7',
    -- '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_classify/model.yt%d.x.oe.t7',
    '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_bid_classify/model.yt%d.x.oe.t7',
    'bi_s100.txt',
    1,
    100)
  --]]
  
  --[[
  generateDataset('/disk/scratch/XingxingZhang/treelstm/experiments/msr/bitreelstm_h400/model_1.0.400.t7', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.sort.h5', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/msr/eot.msr.dep.100.bid.sort.h5')
  --]]
  
  sampleTrees('/disk/scratch/XingxingZhang/treelstm/experiments/msr/bitreelstm_h400/model_1.0.400.t7',
    -- '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_classify/model.yt%d.x.oe.t7',
    -- '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_bid_classify/model.yt%d.x.oe.t7',
    '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_bid_classify_msr/model.yt%d.x.oe.t7',
    'msr_bi_s100.txt',
    1,
    200)
  
end

main()
