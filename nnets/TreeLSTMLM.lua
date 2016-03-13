
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'EMaskedClassNLLCriterion'

require 'basic'

local model_utils = require 'model_utils'

local TreeLSTMLM = torch.class('TreeLSTMLM', 'BModel')

local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function TreeLSTMLM:__init(opts)
  self.opts = opts
  self.name = 'TreeLSTMLM'
  self.rnnCount = 4
  self:print( 'build TreeLSTMLM ...' )
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  opts.seqlen = opts.seqlen or 10
  self.emb, self.lstms, self.softmax, self.softmaxInfer = self:createNetwork(opts)
  self.params, self.grads = model_utils.combine_treelstm_parameters(self.emb, self.lstms, self.softmax)
  
  self.params:uniform(-opts.initRange, opts.initRange)
  self:print( string.format('param size %d\n', self.params:size(1)) )
  print(self.params[{ {1, 10} }])
  
  model_utils.share_lstms_lookuptable(self.emb, self.lstms)
  model_utils.share_linear(self.softmax, self.softmaxInfer)
  
  if opts.wordEmbedding ~= nil and opts.wordEmbedding ~= '' then
    local vocabPath = opts.dataset:sub(1, -3) .. 'vocab.t7'
    model_utils.load_embedding(self.emb, vocabPath, opts.wordEmbedding)
    self:print(string.format('load embeding from %s', opts.wordEmbedding))
  end
  
  self.lstmss = {}
  for i = 1, 4 do
    self.lstmss[i] = model_utils.clone_many_times(self.lstms[i], opts.seqLen)
    self:print(string.format('clone LSTM %d done!', i))
  end
  
  self:print('init states')
  self:initModel(opts)
  self:print('init states done!')
  
  if opts.optimMethod == 'AdaGrad' then
    self.optimMethod = optim.adagrad
  elseif opts.optimMethod == 'Adam' then
    self.optimMethod = optim.adam
  elseif opts.optimMethod == 'AdaDelta' then
    self.optimMethod = optim.adadelta
  elseif opts.optimMethod == 'SGD' then
    self.optimMethod = optim.sgd
  end
  
  self:print( 'build TreeLSTMLM done!' )
end

function TreeLSTMLM:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
  -- compute activations of four gates all together
  local x2h = nn.Linear(nin, nhid * 4)(x_t)
  local h2h = nn.Linear(nhid, nhid * 4)(h_tm1)
  local allGatesActs = nn.CAddTable()({x2h, h2h})
  local allGatesActsSplits = nn.SplitTable(2)( nn.Reshape(4, nhid)(allGatesActs) )
  -- unpack all gate activations
  local i_t = nn.Sigmoid()( nn.SelectTable(1)( allGatesActsSplits ) )
  local f_t = nn.Sigmoid()( nn.SelectTable(2)( allGatesActsSplits ) )
  local o_t = nn.Sigmoid()( nn.SelectTable(3)( allGatesActsSplits ) )
  local n_t = nn.Tanh()( nn.SelectTable(4)( allGatesActsSplits ) )
  -- compute new cell
  local c_t = nn.CAddTable()({
      nn.CMulTable()({ i_t, n_t }),
      nn.CMulTable()({ f_t, c_tm1 })
    })
  -- compute new hidden state
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()( c_t ) })
  
  return c_t, h_t
end

function TreeLSTMLM:createDeepLSTM(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  
  local in_t = {[0] = emb(x_t)}
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  
  local model = nn.gModule({x_t, s_tm1}, {nn.Identity()(s_t)})
  if opts.useGPU then
    return model:cuda()
  else
    return model
  end
end

function TreeLSTMLM:createNetwork(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  if opts.useGPU then emb = emb:cuda() end
  local deepLSTMs = {}
  for i = 1, 4 do 
    table.insert(deepLSTMs, self:createDeepLSTM(opts)) 
  end
  
  local function createSoftmax(inference)
    local h_t = nn.Identity()()
    local y_t = nn.Identity()()
    local div = nn.Identity()()
    
    local dropped = nn.Dropout( opts.dropout )( h_t )
    local h2y = nn.Linear(opts.nhid, opts.novocab)(dropped)
    local y_pred = nn.LogSoftMax()(h2y)
    local err = EMaskedClassNLLCriterion()({y_pred, y_t, div})
    local softmax
    if inference then
      softmax = nn.gModule({h_t, y_t, div}, {err, y_pred})
    else
      softmax = nn.gModule({h_t, y_t, div}, {err})
    end
    
    return softmax
  end
  
  --[[
  local h_t = nn.Identity()()
  local y_t = nn.Identity()()
  local div = nn.Identity()()
  -- local h2y = nn.Linear(opts.nhid, opts.novocab)(h_t)
  local dropped = nn.Dropout( opts.dropout )( h_t )
  local h2y = nn.Linear(opts.nhid, opts.novocab)(dropped)
  local y_pred = nn.LogSoftMax()(h2y)
  local err = EMaskedClassNLLCriterion()({y_pred, y_t, div})
  local softmax = nn.gModule({h_t, y_t, div}, {err})
  --]]
  local softmax = createSoftmax(false)
  local softmaxInfer = createSoftmax(true)
  if opts.useGPU then
    softmax = softmax:cuda()
    softmaxInfer = softmaxInfer:cuda()
  end
  
  return emb, deepLSTMs, softmax, softmaxInfer
end

function TreeLSTMLM:initModel(opts)
  self.embeddings = {}
  self.hiddenStates = {}
  self.df_hiddenStates = {}
  for i = 1, 2 * opts.nlayers do
    self.hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
    self.df_hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
  end
  
  self.initStates = {}
  
  for i = 1, 2*opts.nlayers do
    self.initStates[i] = transferData(opts.useGPU, torch.ones(opts.batchSize, opts.nhid) * opts.initHidVal)
  end
end

function TreeLSTMLM:trainBatch(x, y, sgdParam)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    local T = x:size(1)
    local batchSize = x:size(2)
    for i = 1, 2 * self.opts.nlayers do
      self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
      self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i]
    end
    
    local x_inputs = {}
    local prev_hs = {}
    local cur_hs = {}
    local s_tm1s = {}
    ------------------------------
    -------- forward pass --------
    ------------------------------
    local indices = torch.linspace(1, batchSize, batchSize)
    indices = self.opts.useGPU and indices:cuda() or indices:long()
    for t = 1, T do
      local xs = x[{ t, {}, 1 }]
      local acts = x[{ t, {}, 2 }]
      local prevHs = x[{ t, {}, 3 }]
      local curHs = x[{ t, {}, 4 }]
      
      local x_input = {}
      local prev_h = {}
      local cur_h = {}
      local s_tm1_ = {}
      for act = 1, 4 do
        local lidx = torch.eq(acts, act)
        if lidx:sum() > 0 then
          local bIdx = indices[lidx]
          local x_t = xs[lidx]
          local prevH = (prevHs[lidx] - 1) * batchSize + bIdx
          local curH = (curHs[lidx] - 1) * batchSize + bIdx
          local s_tm1 = {}
          for i = 1, 2*self.opts.nlayers do
            s_tm1[i] = self.hiddenStates[i]:index(1, prevH)
          end
          local s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
          for i = 1, 2*self.opts.nlayers do
            -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
            -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
            self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
          end
          
          x_input[act] = x_t
          prev_h[act] = prevH
          cur_h[act] = curH
          s_tm1_[act] = s_tm1
        else
          x_input[act] = nil
          prev_h[act] = nil
          cur_h[act] = nil
          s_tm1_[act] = nil
        end
      end
      
      x_inputs[t] = x_input
      prev_hs[t] = prev_h
      cur_hs[t] = cur_h
      s_tm1s[t] = s_tm1_
    end
    
    -- now we've got the hidden states self.hiddenStates
    -- ready to compute the softmax
    -- local y_ = y:reshape(y:size(1) * y:size(2))
    local y_ = y:view(y:size(1) * y:size(2))
    local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
    local err = self.softmax:forward({allHiddenStates, y_, batchSize})
    local loss = err
    
    ------------------------------
    -------- backward pass -------
    ------------------------------
    for i = 1, 2 * self.opts.nlayers do
      self.df_hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
      self.df_hiddenStates[i]:zero()
    end
    
    local derr = transferData(self.opts.useGPU, torch.ones(1))
    local df_h_from_y, _, _ = unpack( self.softmax:backward(
      {allHiddenStates, y_, batchSize}, 
      derr
      )
    )
    
    for t = T, 1, -1 do
      for act = 4, 1, -1 do
        if x_inputs[t][act] then
          -- sparseMatrixRowAccOffset(self.df_hiddenStates[2*self.opts.nlayers], cur_hs[t][act], df_h_from_y, batchSize)
          -- sparseMatrixRowAccOffset(self.df_hiddenStates[2*self.opts.nlayers], cur_hs[t][act], df_h_from_y, batchSize)
          local cur_hs_offset = cur_hs[t][act] - batchSize
          local tmp = self.df_hiddenStates[2*self.opts.nlayers]:index(1, cur_hs[t][act]):add(df_h_from_y:index(1, cur_hs_offset))
          self.df_hiddenStates[2*self.opts.nlayers]:indexCopy(1, cur_hs[t][act], tmp)
          
          local d_s_t = {}
          for i = 1, 2*self.opts.nlayers do
            d_s_t[i] = self.df_hiddenStates[i]:index(1, cur_hs[t][act])
          end
          
          local _, d_s_tm1 = unpack(
            self.lstmss[act][t]:backward({ x_inputs[t][act], s_tm1s[t][act] }, d_s_t)
          )
          
          for i = 1, 2*self.opts.nlayers do
            -- sparseMatrixRowAccC(self.df_hiddenStates[i], prev_hs[t][act], d_s_tm1[i])
            -- sparseMatrixRowAccC(self.df_hiddenStates[i], prev_hs[t][act], d_s_tm1[i])
            -- local tmp = self.df_hiddenStates[i]:index(1, prev_hs[t][act]):add(d_s_tm1[i])
            local tmp = self.df_hiddenStates[i]:index(1, prev_hs[t][act]):add(d_s_tm1[i])
            self.df_hiddenStates[i]:indexCopy(1, prev_hs[t][act], tmp)
          end
        end
      end
    end
    
    
    
    -- clip the gradients
    -- self.grads:clamp(-5, 5)
    if self.opts.gradClip < 0 then
      local clip = -self.opts.gradClip
      self.grads:clamp(-clip, clip)
    elseif self.opts.gradClip > 0 then
      local maxGradNorm = self.opts.gradClip
      local gradNorm = self.grads:norm()
      if gradNorm > maxGradNorm then
        local shrinkFactor = maxGradNorm / gradNorm
        self.grads:mul(shrinkFactor)
      end
    end
    
    return loss, self.grads
  end
  
  -- local _, loss_ = optim.adagrad(feval, self.params, sgdParam)
  local _, loss_ = self.optimMethod(feval, self.params, sgdParam)
  return loss_[1]
end

function TreeLSTMLM:validBatch(x, y)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  
  local T = x:size(1)
  local batchSize = x:size(2)
  for i = 1, 2 * self.opts.nlayers do
    self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
    -- self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i]
    self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i][{ {1, batchSize}, {} }]
  end
  
  local x_inputs = {}
  local prev_hs = {}
  local cur_hs = {}
  local s_tm1s = {}
  ------------------------------
  -------- forward pass --------
  ------------------------------
  local indices = torch.linspace(1, batchSize, batchSize)
  indices = self.opts.useGPU and indices:cuda() or indices:long()
  for t = 1, T do
    local xs = x[{ t, {}, 1 }]
    local acts = x[{ t, {}, 2 }]
    local prevHs = x[{ t, {}, 3 }]
    local curHs = x[{ t, {}, 4 }]
    
    local x_input = {}
    local prev_h = {}
    local cur_h = {}
    local s_tm1_ = {}
    for act = 1, 4 do
      local lidx = torch.eq(acts, act)
      if lidx:sum() > 0 then
        local bIdx = indices[lidx]
        local x_t = xs[lidx]
        local prevH = (prevHs[lidx] - 1) * batchSize + bIdx
        local curH = (curHs[lidx] - 1) * batchSize + bIdx
        local s_tm1 = {}
        for i = 1, 2*self.opts.nlayers do
          s_tm1[i] = self.hiddenStates[i]:index(1, prevH)
        end
        local s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
        for i = 1, 2*self.opts.nlayers do
          -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
          -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
          self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
        end
        
        x_input[act] = x_t
        prev_h[act] = prevH
        cur_h[act] = curH
        s_tm1_[act] = s_tm1
      else
        x_input[act] = nil
        prev_h[act] = nil
        cur_h[act] = nil
        s_tm1_[act] = nil
      end
    end
    
    x_inputs[t] = x_input
    prev_hs[t] = prev_h
    cur_hs[t] = cur_h
    s_tm1s[t] = s_tm1_
  end
  
  -- now we've got the hidden states self.hiddenStates
  -- ready to compute the softmax
  local y_ = y:reshape(y:size(1) * y:size(2))
  local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
  local err, y_pred = unpack( self.softmaxInfer:forward({allHiddenStates, y_, batchSize}) )
  local loss = err
  
  return loss, y_pred
  --[[
  local err = self.softmax:forward({allHiddenStates, y_, batchSize})
  local loss = err
  
  return loss
  --]]
end

function TreeLSTMLM:getHiddenStates(x)
  if self.opts.useGPU then
    x = x:cuda()
  end
  
  local T = x:size(1)
  local batchSize = x:size(2)
  for i = 1, 2 * self.opts.nlayers do
    self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
    -- self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i]
    self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i][{ {1, batchSize}, {} }]
  end
  
  local x_inputs = {}
  local prev_hs = {}
  local cur_hs = {}
  local s_tm1s = {}
  ------------------------------
  -------- forward pass --------
  ------------------------------
  local indices = torch.linspace(1, batchSize, batchSize)
  indices = self.opts.useGPU and indices:cuda() or indices:long()
  for t = 1, T do
    local xs = x[{ t, {}, 1 }]
    local acts = x[{ t, {}, 2 }]
    local prevHs = x[{ t, {}, 3 }]
    local curHs = x[{ t, {}, 4 }]
    
    local x_input = {}
    local prev_h = {}
    local cur_h = {}
    local s_tm1_ = {}
    for act = 1, 4 do
      local lidx = torch.eq(acts, act)
      if lidx:sum() > 0 then
        local bIdx = indices[lidx]
        local x_t = xs[lidx]
        local prevH = (prevHs[lidx] - 1) * batchSize + bIdx
        local curH = (curHs[lidx] - 1) * batchSize + bIdx
        local s_tm1 = {}
        for i = 1, 2*self.opts.nlayers do
          s_tm1[i] = self.hiddenStates[i]:index(1, prevH)
        end
        local s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
        for i = 1, 2*self.opts.nlayers do
          -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
          -- sparseMatrixRowAssignC(self.hiddenStates[i], curH, s_t[i])
          self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
        end
        
        x_input[act] = x_t
        prev_h[act] = prevH
        cur_h[act] = curH
        s_tm1_[act] = s_tm1
      else
        x_input[act] = nil
        prev_h[act] = nil
        cur_h[act] = nil
        s_tm1_[act] = nil
      end
    end
    
    x_inputs[t] = x_input
    prev_hs[t] = prev_h
    cur_hs[t] = cur_h
    s_tm1s[t] = s_tm1_
  end
  
  return self.hiddenStates[2*self.opts.nlayers]
end

-- inputs: act, x_t, s_tm1
-- outputs: s_t, y_given_x
function TreeLSTMLM:fpropStep(act, x_t, s_tm1)
  --[[
  print('==in fprop step==')
  print('act')
  print(act)
  print('x_t')
  print(x_t)
  print('s_tm1')
  print(s_tm1)
  print(s_tm1[4][{ 1, {1, 10} }])
  print('cnt')
  print(cnt)
  --]]
  
  local x_t_ = x_t
  local y_t_ = torch.Tensor({0})
  if torch.type(x_t) == 'number' then
    x_t_ = torch.LongTensor({x_t})
    if self.opts.useGPU then 
      x_t_ = x_t_:cuda()
      y_t_ = y_t_:cuda()
    end
  end
  self.lstmss[act][1]:evaluate()
  -- print('x_t_')
  -- print(x_t_)
  local s_t = self.lstmss[act][1]:forward({x_t_, s_tm1})
  local s_t_out = {}
  for i, s in ipairs( s_t ) do
    s_t_out[i] = s:clone()
  end
  self.softmaxInfer:evaluate()
  local err, y_pred = unpack( self.softmaxInfer:forward({s_t[2*self.opts.nlayers], y_t_, 1}) )
  
  return s_t_out, torch.exp(y_pred)
end

function TreeLSTMLM:disableDropout()
  for i = 1, self.rnnCount do
    model_utils.disable_dropout( self.lstmss[i] )
  end
  model_utils.disable_dropout( {self.softmaxInfer} )
end

function TreeLSTMLM:enableDropout()
  for i = 1, self.rnnCount do
    model_utils.enable_dropout( self.lstmss[i] )
  end
  model_utils.enable_dropout( {self.softmaxInfer} )
end



