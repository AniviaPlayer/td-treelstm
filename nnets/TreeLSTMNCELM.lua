
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'EMaskedClassNLLCriterion'
require 'NCE'
require 'NCEMaskedLoss'

require 'basic'

local model_utils = require 'model_utils'

local TreeLSTMNCELM = torch.class('TreeLSTMNCELM', 'BModel')

local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function TreeLSTMNCELM:__init(opts)
  self.opts = opts
  self.name = 'TreeLSTMLMNCE'
  self.rnnCount = 4
  self:print( 'build TreeLSTMLMNCE ...' )
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  opts.seqlen = opts.seqlen or 10
  self.nceZ = torch.exp(opts.lnZ)
  self:print(string.format('lnZ = %f, self.nceZ = %f', opts.lnZ, self.nceZ))
  
  self.emb, self.lstms, self.nce, self.softmax = self:createNetwork(opts)
  self.params, self.grads = model_utils.combine_treelstm_parameters(self.emb, self.lstms, self.nce)
  
  self.params:uniform(-opts.initRange, opts.initRange)
  self:print( string.format('param size %d\n', self.params:size(1)) )
  print(self.params[{ {1, 10} }])
  
  model_utils.share_lstms_lookuptable(self.emb, self.lstms)
  local nce_module, _ = model_utils.share_nce_softmax(self.nce, self.softmax)
  if opts.learnZ then
    nce_module.bias:fill(self.nceZ)
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
  
  self:print( 'build TreeLSTMLMNCE done!' )
end

function TreeLSTMNCELM:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
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

function TreeLSTMNCELM:createDeepLSTM(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  local x_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  
  local in_t = {[0] = emb(x_t)}
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    -- local x_t_i = in_t[i - 1]
    local x_t_i = nn.Dropout( opts.dropout )( in_t[i - 1] )
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

function TreeLSTMNCELM:createNetwork(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  if opts.useGPU then emb = emb:cuda() end
  local deepLSTMs = {}
  for i = 1, 4 do 
    table.insert(deepLSTMs, self:createDeepLSTM(opts)) 
  end
  
  local function createNCE()
    local h_t = nn.Identity()()
    local y_t = nn.Identity()()
    local mask_t = nn.Identity()()
    local y_neg_t = nn.Identity()()
    local y_prob_t = nn.Identity()()
    local y_neg_prob_t = nn.Identity()()
    local div = nn.Identity()()
    
    local dropped = nn.Dropout( opts.dropout )( h_t )
    local nce_cost = NCE(opts.nhid, opts.novocab, self.nceZ, opts.learnZ)({dropped, y_t,
        y_neg_t, y_prob_t, y_neg_prob_t})
    local nce_loss = NCEMaskedLoss()({nce_cost, mask_t, div})
    local nce = nn.gModule({h_t, y_t, y_neg_t, y_prob_t, y_neg_prob_t, mask_t, div}, {nce_loss})
    
    return nce
  end
  
  local function createSoftmax()
    local h_t = nn.Identity()()
    local y_t = nn.Identity()()
    local div = nn.Identity()()
    
    local dropped = nn.Dropout( opts.dropout )( h_t )
    local h2y = nn.Linear(opts.nhid, opts.novocab)(dropped)
    local y_pred = nn.LogSoftMax()(h2y)
    local err = EMaskedClassNLLCriterion()({y_pred, y_t, div})
    local softmax = nn.gModule({h_t, y_t, div}, {err, y_pred})
    
    return softmax
  end
  
  local nce = createNCE()
  local softmax = createSoftmax()
  
  if opts.useGPU then
    nce = nce:cuda()
    softmax = softmax:cuda()
  end
  
  return emb, deepLSTMs, nce, softmax
end

function TreeLSTMNCELM:initModel(opts)
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

-- inputs: x, y, y_neg, y_prob, y_neg_prob, mask
function TreeLSTMNCELM:trainBatch(x, y, y_neg, y_prob, y_neg_prob, mask, sgdParam)
  if self.opts.useGPU then
    x = x:cuda()
    mask = mask:cuda()
    y = y:cuda()
    y_neg = y_neg:cuda()
    y_prob = y_prob:cuda()
    y_neg_prob = y_neg_prob:cuda()
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
    
    -- print('compute LSTM states done!')
    
    -- now we've got the hidden states self.hiddenStates
    -- ready to compute the softmax
    local y_ = y:view(y:size(1) * y:size(2))
    local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
    -- local err = self.softmax:forward({allHiddenStates, y_, batchSize})
    -- {h_t, y_t, y_neg_t, y_prob_t, y_neg_prob_t, mask_t, div}
    local y_neg_ = y_neg:view(y_neg:size(1) * y_neg:size(2), y_neg:size(3))
    local y_prob_ = y_prob:view(-1)
    local y_neg_prob_ = y_neg_prob:view(y_neg_prob:size(1) * y_neg_prob:size(2), y_neg_prob:size(3))
    
    -- print('before doing nce')
    
    local err = self.nce:forward({allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize})
    
    -- printf('NCE forward done! loss = %f\n', err)
    local loss = err
    
    ------------------------------
    -------- backward pass -------
    ------------------------------
    for i = 1, 2 * self.opts.nlayers do
      self.df_hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
      self.df_hiddenStates[i]:zero()
    end
    
    local derr = transferData(self.opts.useGPU, torch.ones(1))
    --[[
    local df_h_from_y, _, _ = unpack( self.softmax:backward(
      {allHiddenStates, y_, batchSize}, 
      derr
      )
    )
    --]]
    -- print('before nce backward pass')
    
    local df_h_from_y, _, _, _, _, _, _ = unpack( self.nce:backward(
      {allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize}, 
      derr
      )
    )
    
    --[[
    print('backward pass done')
    print('df_h size')
    print(df_h_from_y:size())
    --]]
    
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

function TreeLSTMNCELM:validBatch(x, y)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  
  local T = x:size(1)
  local batchSize = x:size(2)
  for i = 1, 2 * self.opts.nlayers do
    self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
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
  local err, y_pred = unpack( self.softmax:forward({allHiddenStates, y_, batchSize}) )
  local loss = err
  
  return loss, y_pred
end

function TreeLSTMNCELM:disableDropout()
  for i = 1, self.rnnCount do
    model_utils.disable_dropout( self.lstmss[i] )
  end
  model_utils.disable_dropout( {self.softmax} )
end

function TreeLSTMNCELM:enableDropout()
  for i = 1, self.rnnCount do
    model_utils.enable_dropout( self.lstmss[i] )
  end
  model_utils.enable_dropout( {self.softmax} )
end

