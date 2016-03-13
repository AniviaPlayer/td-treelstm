
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'EMaskedClassNLLCriterion'
require 'NCE'
require 'NCEMaskedLoss'

require 'basic'

local model_utils = require 'model_utils'

local BiTreeLSTMNCELM = torch.class('BiTreeLSTMNCELM', 'BModel')

-- actions = {JL = 1, JR = 2, JLF = 3, JRF = 4}
local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function BiTreeLSTMNCELM:__init(opts)
  self.opts = opts
  self.name = 'BiTreeLSTMNCELM'
  self.rnnCount = 5
  self:print( 'build BiTreeLSTMNCELM ...' )
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
  
  if opts.wordEmbedding ~= nil and opts.wordEmbedding ~= '' then
    local vocabPath = opts.dataset:sub(1, -3) .. 'vocab.t7'
    model_utils.load_embedding(self.emb, vocabPath, opts.wordEmbedding)
    self:print(string.format('load embeding from %s', opts.wordEmbedding))
  end
  
  self.lstmss = {}
  for i = 1, self.rnnCount do
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
  
  self:print( 'build BiTreeLSTMNCELM done!' )
end

function BiTreeLSTMNCELM:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
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

function BiTreeLSTMNCELM:createDeepLSTMContext(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  local x_t = nn.Identity()()
  local cxt_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  -- join word embedings of x_t and context vector
  local x_cxt_t = nn.JoinTable(2){emb(x_t), cxt_t}
  local in_t = {[0] = x_cxt_t}
  local s_t = {}
  local splits_tm1 = {s_tm1:split(2 * opts.nlayers)}
  
  for i = 1, opts.nlayers do
    local c_tm1_i = splits_tm1[i + i - 1]
    local h_tm1_i = splits_tm1[i + i]
    local x_t_i = in_t[i - 1]
    local c_t_i, h_t_i = nil, nil
    if i == 1 then
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nin + opts.nlchid, opts.nhid)
    else
      c_t_i, h_t_i = self:createLSTM(x_t_i, c_tm1_i, h_tm1_i, opts.nhid, opts.nhid)
    end
    s_t[i+i-1] = c_t_i
    s_t[i+i] = h_t_i
    in_t[i] = h_t_i
  end
  
  local model = nn.gModule({x_t, cxt_t, s_tm1}, {nn.Identity()(s_t)})
  if opts.useGPU then
    return model:cuda()
  else
    return model
  end
end

function BiTreeLSTMNCELM:createDeepLSTM(opts)
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

function BiTreeLSTMNCELM:createNetwork(opts)
  local emb = nn.LookupTable(opts.nivocab, opts.nin)
  if opts.useGPU then emb = emb:cuda() end
  local deepLSTMs = {}
  deepLSTMs[1] = self:createDeepLSTM(opts)
  -- actions = {JL = 1, JR = 2, JLF = 3, JRF = 4}
  deepLSTMs[2] = self:createDeepLSTMContext(opts)
  deepLSTMs[3] = self:createDeepLSTM(opts)
  deepLSTMs[4] = self:createDeepLSTM(opts)
  print 'create 4 lstms done!'
  -- yet another LSTM to learning left children represenations
  local lc_opts = {nivocab = opts.nivocab, nin = opts.nin, nlayers = opts.nlclayers, nhid = opts.nlchid, useGPU = opts.useGPU}
  print(lc_opts)
  deepLSTMs[5] = self:createDeepLSTM(lc_opts)
  print 'create left child lstm done!'
  
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

function BiTreeLSTMNCELM:initModel(opts)
  self.embeddings = {}
  self.hiddenStates = {}
  self.df_hiddenStates = {}
  -- self.lc_hiddenStates = {}
  -- self.df_lc_hiddenStates = {}
  for i = 1, 2 * opts.nlayers do
    self.hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
    self.df_hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
    -- self.lc_hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
    -- self.df_lc_hiddenStates[i] = opts.useGPU and torch.CudaTensor() or torch.Tensor()
  end
  
  self.initStates = {}
  -- for simplificity, I assume init states
  -- for left child RNN and TreeRNN is the same
  -- self.lc_initStates = {}
  
  for i = 1, 2*opts.nlayers do
    self.initStates[i] = transferData(opts.useGPU, torch.ones(opts.batchSize, opts.nhid) * opts.initHidVal)
  end
  
  self.lc_initStates = {}
  for i = 1, 2*opts.nlclayers do
    self.lc_initStates[i] = transferData(opts.useGPU, torch.ones(opts.batchSize, opts.nlchid) * opts.initHidVal)
  end
  -- last hidden layer of the left children RNN
  self.lc_hsL = opts.useGPU and torch.CudaTensor() or torch.Tensor()
  self.df_lc_hsL = opts.useGPU and torch.CudaTensor() or torch.Tensor()
end
-- y_neg, y_prob, y_neg_prob, mask, 
function BiTreeLSTMNCELM:trainBatch(x, y, lc, lc_mask, y_neg, y_prob, y_neg_prob, mask, sgdParam)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
    lc = lc:cuda()
    lc_mask = lc_mask:cuda()
    y_neg = y_neg:cuda()
    y_prob = y_prob:cuda()
    y_neg_prob = y_neg_prob:cuda()
    mask = mask:cuda()
  end
  
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    
    --------------------------------------------
    -- forward pass for left child RNN
    local lcT, lcBatchSize, lc_hiddenStates, lc_hs0
    
    if lc:dim() ~= 0 then
      lcT = lc:size(1)
      lcBatchSize = lc:size(2)
    else
      lcT = 0
      lcBatchSize = 1
    end
    lc_hiddenStates = {}
    lc_hs0 = {}
    
    for i = 1, 2 * self.opts.nlclayers do
      table.insert( lc_hs0, self.lc_initStates[i][{ {1, lcBatchSize}, {} }] )
    end
    lc_hiddenStates[0] = lc_hs0
    self.lc_hsL:resize((lcT + 1) * lcBatchSize, self.opts.nlchid)
    self.lc_hsL[{ {1, lcBatchSize}, {} }] = lc_hs0[2 * self.opts.nlclayers]
    
    if lc:dim() ~= 0 then
      for t = 1, lcT do
        -- this is for flush
        if t ~= 1 then
          local mask_ = lc_mask[{ t, {} }]:eq(2) -- this is the begining of a new sub-sequence
          if mask_:sum() > 0 then
            local idxs = mask_:float():nonzero():view(-1)
            if self.opts.useGPU then idxs = idxs:cuda() end
            for i = 1, 2*self.opts.nlclayers do
              lc_hiddenStates[t-1][i]:indexCopy(1, idxs, self.initStates[i][{ {1, idxs:size(1)}, {} }])
            end
          end
        end -- end of flush
        lc_hiddenStates[t] = self.lstmss[5][t]:forward({lc[{ t, {} }], lc_hiddenStates[t - 1]})
        self.lc_hsL[{ {t*lcBatchSize+1, (t+1)*lcBatchSize}, {} }] = lc_hiddenStates[t][2*self.opts.nlclayers]
      end
    end
    -- done with forward pass for left child RNN
    ---------------------------------------------
    
    local T = x:size(1)
    local batchSize = x:size(2)
    for i = 1, 2 * self.opts.nlayers do
      self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
      self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i][{ {1, batchSize}, {} }]
    end
    
    -- temps for faster implmentation
    -- no re-compution of these value during backprop
    local x_inputs = {}
    local prev_hs = {}
    local cur_hs = {}
    local s_tm1s = {}
    -- this temp is for GEN-R RNN, to put context learned by lc RNN
    local lc_cxts = {}
    local lc_icxts = {}
    
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
      local lc_cxt = {} -- this temp is for GEN-R RNN, to put context learned by lc RNN
      local lc_icxt = {}
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
          local s_t
          local lcc -- this temp is for GEN-R RNN, to put context learned by lc RNN
          local lcic
          if act ~= 2 then
            lcic = nil
            lcc = nil
            s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
          else -- for GEN-R RNN; need LC RNN addtional input
            lcic = x[{ t, {}, 5 }][lidx] + lcBatchSize
            lcc = self.lc_hsL:index(1, lcic)
            s_t = self.lstmss[act][t]:forward({ x_t, lcc, s_tm1 })
          end
          for i = 1, 2*self.opts.nlayers do
            self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
          end
          
          x_input[act] = x_t
          prev_h[act] = prevH
          cur_h[act] = curH
          s_tm1_[act] = s_tm1
          lc_cxt[act] = lcc
          lc_icxt[act] = lcic
        else
          x_input[act] = nil
          prev_h[act] = nil
          cur_h[act] = nil
          s_tm1_[act] = nil
          lc_cxt[act] = nil
          lc_icxt[act] = nil
        end
      end
      
      x_inputs[t] = x_input
      prev_hs[t] = prev_h
      cur_hs[t] = cur_h
      s_tm1s[t] = s_tm1_
      lc_cxts[t] = lc_cxt
      lc_icxts[t] = lc_icxt
    end
    
    --[[
    -- now we've got the hidden states self.hiddenStates
    -- ready to compute the softmax
    -- local y_ = y:reshape(y:size(1) * y:size(2))
    local y_ = y:view(y:size(1) * y:size(2))
    local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
    local err = self.softmax:forward({allHiddenStates, y_, batchSize})
    local loss = err
    --]]
    
    -- now we've got the hidden states self.hiddenStates
    -- ready to compute the nce
    local y_ = y:view(y:size(1) * y:size(2))
    local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
    -- local err = self.softmax:forward({allHiddenStates, y_, batchSize})
    -- {h_t, y_t, y_neg_t, y_prob_t, y_neg_prob_t, mask_t, div}
    local y_neg_ = y_neg:view(y_neg:size(1) * y_neg:size(2), y_neg:size(3))
    local y_prob_ = y_prob:view(-1)
    local y_neg_prob_ = y_neg_prob:view(y_neg_prob:size(1) * y_neg_prob:size(2), y_neg_prob:size(3))
    local err = self.nce:forward({allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize})
    
    local loss = err
    
    -- preprare bp for left child RNN
    self.df_lc_hsL:resize((lcT + 1) * lcBatchSize, self.opts.nlchid):zero()
    
    ------------------------------
    -------- backward pass -------
    ------------------------------
    
    --[[
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
    --]]
    
    for i = 1, 2 * self.opts.nlayers do
      self.df_hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
      self.df_hiddenStates[i]:zero()
    end
    
    local derr = transferData(self.opts.useGPU, torch.ones(1))
    
    local df_h_from_y, _, _, _, _, _, _ = unpack( self.nce:backward(
      {allHiddenStates, y_, y_neg_, y_prob_, y_neg_prob_, mask, batchSize}, 
      derr
      )
    )
    
    for t = T, 1, -1 do
      for act = 4, 1, -1 do
        if x_inputs[t][act] then
          local cur_hs_offset = cur_hs[t][act] - batchSize
          local tmp = self.df_hiddenStates[2*self.opts.nlayers]:index(1, cur_hs[t][act]):add(df_h_from_y:index(1, cur_hs_offset))
          self.df_hiddenStates[2*self.opts.nlayers]:indexCopy(1, cur_hs[t][act], tmp)
          
          local d_s_t = {}
          for i = 1, 2*self.opts.nlayers do
            d_s_t[i] = self.df_hiddenStates[i]:index(1, cur_hs[t][act])
          end
          
          local _d_x_inputs
          local d_s_tm1
          local d_lc_cxt
          if act ~= 2 then
            _d_x_inputs, d_s_tm1 = unpack(
              self.lstmss[act][t]:backward({ x_inputs[t][act], s_tm1s[t][act] }, d_s_t)
            )
            d_lc_cxt = nil
          else  -- this temp is for GEN-R RNN, to put context learned by lc RNN
            _d_x_inputs, d_lc_cxt, d_s_tm1 = unpack(
              self.lstmss[act][t]:backward({ x_inputs[t][act], lc_cxts[t][act], s_tm1s[t][act] }, d_s_t)
            )
            -- copy errors back to df_lc_hsL
            local tmp = self.df_lc_hsL:index(1, lc_icxts[t][act]):add(d_lc_cxt)
            self.df_lc_hsL:indexCopy(1, lc_icxts[t][act], tmp)
          end
          
          for i = 1, 2*self.opts.nlayers do
            local tmp = self.df_hiddenStates[i]:index(1, prev_hs[t][act]):add(d_s_tm1[i])
            self.df_hiddenStates[i]:indexCopy(1, prev_hs[t][act], tmp)
          end
        end
      end
    end
    
    ---------------------------------------------
    -- backward pass for left child RNN
    if lc:dim() ~= 0 then
      local df_lc_hsT = {}
      for i = 1, 2 * self.opts.nlclayers do
        local tmp = self.opts.useGPU and torch.CudaTensor() or torch.Tensor()
        tmp:resize(lcBatchSize, self.opts.nlchid):zero()
        table.insert( df_lc_hsT, tmp )
      end
      local df_lc_hiddenStates = {[lcT] = df_lc_hsT}
      for t = lcT, 1, -1 do
        local mask_ = lc_mask[{ t, {} }]:eq(0)
        if mask_:sum() > 0 then
          local idxs = mask_:float():nonzero():view(-1)
          if self.opts.useGPU then idxs = idxs:cuda() end
          for i = 1, 2*self.opts.nlclayers do
            df_lc_hiddenStates[t][i]:indexFill(1, idxs, 0)
          end
        end
        df_lc_hiddenStates[t][2 * self.opts.nlclayers]:add( self.df_lc_hsL[{ {t*lcBatchSize+1, (t+1)*lcBatchSize}, {} }] )
        local tmp
        tmp, df_lc_hiddenStates[t - 1] = unpack(
          self.lstmss[5][t]:backward( {lc[{ t, {} }], lc_hiddenStates[t - 1]}, df_lc_hiddenStates[t] )
        )
        tmp = nil
        
        -- this is for flush
        if t > 1 then
          local mask_ = lc_mask[{ t, {} }]:eq(2) -- 2: start of a sequence; 0: padded
          if mask_:sum() > 0 then
            local idxs = mask_:float():nonzero():view(-1)
            if self.opts.useGPU then idxs = idxs:cuda() end
            for i = 1, 2*self.opts.nlclayers do
              df_lc_hiddenStates[t-1][i]:indexFill(1, idxs, 0)
            end
          end
        end -- end of flush gradients
      end
    end
    -- done with backward pass for left child RNN
    ---------------------------------------------
    
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

function BiTreeLSTMNCELM:validBatch(x, y, lc, lc_mask)
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
    lc = lc:cuda()
    lc_mask = lc_mask:cuda()
  end
  
  --------------------------------------------
  -- forward pass for left child RNN
  local lcT, lcBatchSize, lc_hiddenStates, lc_hs0
  
  if lc:dim() ~= 0 then
    lcT = lc:size(1)
    lcBatchSize = lc:size(2)
  else
    lcT = 0
    lcBatchSize = 1
  end
  lc_hiddenStates = {}
  lc_hs0 = {}
  
  for i = 1, 2 * self.opts.nlclayers do
    table.insert( lc_hs0, self.lc_initStates[i][{ {1, lcBatchSize}, {} }] )
  end
  
  lc_hiddenStates[0] = lc_hs0
  self.lc_hsL:resize((lcT + 1) * lcBatchSize, self.opts.nlchid)
  self.lc_hsL[{ {1, lcBatchSize}, {} }] = lc_hs0[2 * self.opts.nlclayers]
  
  if lc:dim() ~= 0 then
    for t = 1, lcT do
      -- this is for flush
      if t ~= 1 then
        local mask_ = lc_mask[{ t, {} }]:eq(2) -- this is the begining of a new sub-sequence
        if mask_:sum() > 0 then
          local idxs = mask_:float():nonzero():view(-1)
          if self.opts.useGPU then idxs = idxs:cuda() end
          for i = 1, 2*self.opts.nlclayers do
            lc_hiddenStates[t-1][i]:indexCopy(1, idxs, self.initStates[i][{ {1, idxs:size(1)}, {} }])
          end
        end
      end -- end of flush
      lc_hiddenStates[t] = self.lstmss[5][t]:forward({lc[{ t, {} }], lc_hiddenStates[t - 1]})
      self.lc_hsL[{ {t*lcBatchSize+1, (t+1)*lcBatchSize}, {} }] = lc_hiddenStates[t][2*self.opts.nlclayers]
    end
  end
  -- done with forward pass for left child RNN
  ---------------------------------------------
  
  local T = x:size(1)
  local batchSize = x:size(2)
  for i = 1, 2 * self.opts.nlayers do
    self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
    self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i][{ {1, batchSize}, {} }]
  end
  
  -- temps for faster implmentation
  -- no re-compution of these value during backprop
  local x_inputs = {}
  local prev_hs = {}
  local cur_hs = {}
  local s_tm1s = {}
  -- this temp is for GEN-R RNN, to put context learned by lc RNN
  local lc_cxts = {}
  local lc_icxts = {}
  
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
    local lc_cxt = {} -- this temp is for GEN-R RNN, to put context learned by lc RNN
    local lc_icxt = {}
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
        local s_t
        local lcc -- this temp is for GEN-R RNN, to put context learned by lc RNN
        local lcic
        if act ~= 2 then
          lcic = nil
          lcc = nil
          s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
        else -- for GEN-R RNN; need LC RNN addtional input
          lcic = x[{ t, {}, 5 }][lidx] + lcBatchSize
          lcc = self.lc_hsL:index(1, lcic)
          s_t = self.lstmss[act][t]:forward({ x_t, lcc, s_tm1 })
        end
        for i = 1, 2*self.opts.nlayers do
          self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
        end
        
        x_input[act] = x_t
        prev_h[act] = prevH
        cur_h[act] = curH
        s_tm1_[act] = s_tm1
        lc_cxt[act] = lcc
        lc_icxt[act] = lcic
      else
        x_input[act] = nil
        prev_h[act] = nil
        cur_h[act] = nil
        s_tm1_[act] = nil
        lc_cxt[act] = nil
        lc_icxt[act] = nil
      end
    end
    
    x_inputs[t] = x_input
    prev_hs[t] = prev_h
    cur_hs[t] = cur_h
    s_tm1s[t] = s_tm1_
    lc_cxts[t] = lc_cxt
    lc_icxts[t] = lc_icxt
  end
  
  -- now we've got the hidden states self.hiddenStates
  -- ready to compute the softmax
  local y_ = y:reshape(y:size(1) * y:size(2))
  local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
  local err, y_pred = unpack( self.softmax:forward({allHiddenStates, y_, batchSize}) )
  local loss = err
  
  return loss, y_pred
end

function BiTreeLSTMNCELM:getHiddenStates(x, lc, lc_mask)
  if self.opts.useGPU then
    x = x:cuda()
    -- y = y:cuda()
    lc = lc:cuda()
    lc_mask = lc_mask:cuda()
  end
  
  --------------------------------------------
  -- forward pass for left child RNN
  local lcT, lcBatchSize, lc_hiddenStates, lc_hs0
  
  if lc:dim() ~= 0 then
    lcT = lc:size(1)
    lcBatchSize = lc:size(2)
  else
    lcT = 0
    lcBatchSize = 1
  end
  lc_hiddenStates = {}
  lc_hs0 = {}
  
  for i = 1, 2 * self.opts.nlclayers do
    table.insert( lc_hs0, self.lc_initStates[i][{ {1, lcBatchSize}, {} }] )
  end
  
  lc_hiddenStates[0] = lc_hs0
  self.lc_hsL:resize((lcT + 1) * lcBatchSize, self.opts.nlchid)
  self.lc_hsL[{ {1, lcBatchSize}, {} }] = lc_hs0[2 * self.opts.nlclayers]
  
  if lc:dim() ~= 0 then
    for t = 1, lcT do
      -- this is for flush
      if t ~= 1 then
        local mask_ = lc_mask[{ t, {} }]:eq(2) -- this is the begining of a new sub-sequence
        if mask_:sum() > 0 then
          local idxs = mask_:float():nonzero():view(-1)
          if self.opts.useGPU then idxs = idxs:cuda() end
          for i = 1, 2*self.opts.nlclayers do
            lc_hiddenStates[t-1][i]:indexCopy(1, idxs, self.initStates[i][{ {1, idxs:size(1)}, {} }])
          end
        end
      end -- end of flush
      lc_hiddenStates[t] = self.lstmss[5][t]:forward({lc[{ t, {} }], lc_hiddenStates[t - 1]})
      self.lc_hsL[{ {t*lcBatchSize+1, (t+1)*lcBatchSize}, {} }] = lc_hiddenStates[t][2*self.opts.nlclayers]
    end
  end
  -- done with forward pass for left child RNN
  ---------------------------------------------
  
  local T = x:size(1)
  local batchSize = x:size(2)
  for i = 1, 2 * self.opts.nlayers do
    self.hiddenStates[i]:resize((T + 1) * batchSize, self.opts.nhid)
    self.hiddenStates[i][{ {1, batchSize}, {} }] = self.initStates[i][{ {1, batchSize}, {} }]
  end
  
  -- temps for faster implmentation
  -- no re-compution of these value during backprop
  local x_inputs = {}
  local prev_hs = {}
  local cur_hs = {}
  local s_tm1s = {}
  -- this temp is for GEN-R RNN, to put context learned by lc RNN
  local lc_cxts = {}
  local lc_icxts = {}
  
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
    local lc_cxt = {} -- this temp is for GEN-R RNN, to put context learned by lc RNN
    local lc_icxt = {}
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
        local s_t
        local lcc -- this temp is for GEN-R RNN, to put context learned by lc RNN
        local lcic
        if act ~= 2 then
          lcic = nil
          lcc = nil
          s_t = self.lstmss[act][t]:forward({ x_t, s_tm1 })
        else -- for GEN-R RNN; need LC RNN addtional input
          lcic = x[{ t, {}, 5 }][lidx] + lcBatchSize
          lcc = self.lc_hsL:index(1, lcic)
          s_t = self.lstmss[act][t]:forward({ x_t, lcc, s_tm1 })
        end
        for i = 1, 2*self.opts.nlayers do
          self.hiddenStates[i]:indexCopy(1, curH, s_t[i])
        end
        
        x_input[act] = x_t
        prev_h[act] = prevH
        cur_h[act] = curH
        s_tm1_[act] = s_tm1
        lc_cxt[act] = lcc
        lc_icxt[act] = lcic
      else
        x_input[act] = nil
        prev_h[act] = nil
        cur_h[act] = nil
        s_tm1_[act] = nil
        lc_cxt[act] = nil
        lc_icxt[act] = nil
      end
    end
    
    x_inputs[t] = x_input
    prev_hs[t] = prev_h
    cur_hs[t] = cur_h
    s_tm1s[t] = s_tm1_
    lc_cxts[t] = lc_cxt
    lc_icxts[t] = lc_icxt
  end
  
  --[[
  -- now we've got the hidden states self.hiddenStates
  -- ready to compute the softmax
  local y_ = y:reshape(y:size(1) * y:size(2))
  local allHiddenStates = self.hiddenStates[2*self.opts.nlayers][{ {batchSize + 1, -1}, {} }]
  local err, y_pred = unpack( self.softmaxInfer:forward({allHiddenStates, y_, batchSize}) )
  local loss = err
  
  return loss, y_pred
  --]]
  
  return self.hiddenStates[2*self.opts.nlayers], self.lc_hsL
end

function BiTreeLSTMNCELM:fpropStep(act, x_t, s_tm1, lcs)
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
  
  local s_t
  if act ~= 2 then
    s_t = self.lstmss[act][1]:forward({x_t_, s_tm1})
  else
    -- run left child RNN
    local lc_hs0 = {}
    for i = 1, 2 * self.opts.nlclayers do
      table.insert( lc_hs0, self.lc_initStates[i][{ {1}, {} }] )
    end
    local lc_hids = {[0] = lc_hs0}
    local T = #lcs
    for t = 1, T do
      local lc = torch.LongTensor({lcs[t]})
      if self.opts.useGPU then lc = lc:cuda() end
      lc_hids[t] = self.lstmss[5][t]:forward({ lc, lc_hids[t-1] })
    end
    local lcc = lc_hids[T][2 * self.opts.nlclayers]
    s_t = self.lstmss[act][1]:forward({x_t_, lcc, s_tm1})
  end
  
  local s_t_out = {}
  for i, s in ipairs( s_t ) do
    s_t_out[i] = s:clone()
  end
  self.softmax:evaluate()
  local err, y_pred = unpack( self.softmax:forward({s_t[2*self.opts.nlayers], y_t_, 1}) )
  
  return s_t_out, torch.exp(y_pred)
end

function BiTreeLSTMNCELM:disableDropout()
  for i = 1, self.rnnCount do
    model_utils.disable_dropout( self.lstmss[i] )
  end
  model_utils.disable_dropout( {self.softmax} )
end

function BiTreeLSTMNCELM:enableDropout()
  for i = 1, self.rnnCount do
    model_utils.enable_dropout( self.lstmss[i] )
  end
  model_utils.enable_dropout( {self.softmax} )
end
