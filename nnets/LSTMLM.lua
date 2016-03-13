
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'Embedding'
require 'MaskedClassNLLCriterion'

local model_utils = require 'model_utils'

local LSTMLM = torch.class('LSTMLM')

function LSTMLM:__init(opts)
  print 'build LSTMLM ...'
  torch.manualSeed(opts.seed)
  -- build model
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  opts.seqlen = opts.seqlen or 10
  self.emb = Embedding(opts.nivocab, opts.nin)
  -- self.lstm = self:createLSTM(opts.nin, opts.nhid)
  print 'faster LSTM implmentation?'
  self.lstm = self:createLSTMFaster(opts.nin, opts.nhid)

  self.softmax = nn.Sequential():add( nn.Linear(opts.nhid, opts.novocab) ):add( nn.LogSoftMax() )
  self.params, self.grads = model_utils.combine_all_parameters(self.emb, self.lstm, self.softmax)
  print('init range', -opts.initRange, opts.initRange)
  self.params:uniform(-opts.initRange, opts.initRange)

  -- clone everything --
  print 'begin to clone emb'
  self.embs = model_utils.clone_many_times(self.emb, opts.seqLen)
  print 'clone emb done'
  print 'begin to clone lstm'
  self.lstms = model_utils.clone_many_times(self.lstm, opts.seqLen)
  print 'clone lstm done!'
  print 'begin to clone softmax'
  self.softmaxs = model_utils.clone_many_times(self.softmax, opts.seqLen)
  print 'clone softmax done!'
  
  self.h0s = torch.ones(opts.batchSize, opts.nhid) * opts.initHidVal
  self.c0s = self.h0s:clone()
  self.d_hTs = torch.zeros(opts.batchSize, opts.nhid)
  self.d_cTs = self.d_hTs:clone()

  -- self.criterion = nn.ClassNLLCriterion()
  self.criterion = MaskedClassNLLCriterion()
  self.criterions = model_utils.clone_many_times(self.criterion, opts.seqLen)
  print 'build LSTMLM done!'
end

function LSTMLM:createLSTM(nin, nhid)
  -- inputs
  local x_t = nn.Identity()()
  local c_tm1 = nn.Identity()()
  local h_tm1 = nn.Identity()()

  local function newHidLinear()
    local i2h = nn.Linear(nin, nhid)(x_t)
    local h2h = nn.Linear(nhid, nhid)(h_tm1)

    return nn.CAddTable()({i2h, h2h})
  end

  local i_t = nn.Sigmoid()( newHidLinear() )
  local f_t = nn.Sigmoid()( newHidLinear() )
  local o_t = nn.Sigmoid()( newHidLinear() )
  local n_t = nn.Tanh()( newHidLinear() )

  local c_t = nn.CAddTable()({
    nn.CMulTable()({ f_t,  c_tm1 }),
    nn.CMulTable()({ i_t, n_t })
  })

  local h_t = nn.CMulTable()({ o_t, nn.Tanh()(c_t) })

  return nn.gModule({x_t, c_tm1, h_tm1}, {c_t, h_t})
end

function LSTMLM:createLSTMFaster(nin, nhid)
  local x_t = nn.Identity()()
  local c_tm1 = nn.Identity()()
  local h_tm1 = nn.Identity()()
  -- compute four gates together
  local x2h = nn.Linear(nin, 4*nhid)(x_t)
  local h2h = nn.Linear(nhid, 4*nhid)(h_tm1)
  local gateActs = nn.CAddTable()({x2h, h2h})
  -- split the activations of four gates into four
  local reshapedGateActs = nn.Reshape(4, nhid)(gateActs)
  local gateActsSplits = nn.SplitTable(2)(reshapedGateActs)
  -- unpack all gates
  local i_t = nn.Sigmoid()( nn.SelectTable(1)(gateActsSplits) )
  local f_t = nn.Sigmoid()( nn.SelectTable(2)(gateActsSplits) )
  local o_t = nn.Sigmoid()( nn.SelectTable(3)(gateActsSplits) )
  local n_t = nn.Tanh()( nn.SelectTable(4)(gateActsSplits) )
  
  local c_t = nn.CAddTable()({
      nn.CMulTable()({i_t, n_t}),
      nn.CMulTable()({f_t, c_tm1})
    })
  local h_t = nn.CMulTable()({ o_t, nn.Tanh()(c_t) })
  
  return nn.gModule({x_t, c_tm1, h_tm1}, {c_t, h_t})
end

function LSTMLM:validBatch(x, y)
  local bs = x:size(2)
  local embeds = {}
  local hs = {[0] = self.h0s[{ {1, bs}, {} }]}
  local cs = {[0] = self.c0s[{ {1, bs}, {} }]}
  local log_y_preds= {}
  local loss = 0
  local T = x:size(1)
  for t = 1, T do
    embeds[t] = self.embs[t]:forward(x[{ t, {} }])
    cs[t], hs[t] = unpack(
      self.lstms[t]:forward({embeds[t], cs[t-1], hs[t-1]})
    )
    log_y_preds[t] = self.softmaxs[t]:forward(hs[t])
    loss = loss + self.criterions[t]:forward(log_y_preds[t], y[{ t, {} }])
  end
  
  return loss
end

function LSTMLM:trainBatch(x, y, sgd_param)
  -- x: (seqlen, bs)
  -- y: (seqlen, bs)
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    
    self.grads:zero()
    -- forward pass
    local bs = x:size(2)
    local embeds = {}
    local hs = {[0] = self.h0s[{ {1, bs}, {} }]}
    local cs = {[0] = self.c0s[{ {1, bs}, {} }]}
    local log_y_preds= {}
    local loss = 0
    local T = x:size(1)
    for t = 1, T do
      embeds[t] = self.embs[t]:forward(x[{ t, {} }])
      cs[t], hs[t] = unpack(
        self.lstms[t]:forward({embeds[t], cs[t-1], hs[t-1]})
      )
      log_y_preds[t] = self.softmaxs[t]:forward(hs[t])
      loss = loss + self.criterions[t]:forward(log_y_preds[t], y[{ t, {} }])
    end

    -- backward pass
    local d_embeds = {}
    self.d_hTs:zero()
    local d_hs = {[T] = self.d_hTs[{ {1, bs}, {} }]}
    -- local d_hs = {}
    local d_cs = {[T] = self.d_cTs[{ {1, bs}, {} }]}
    for t = T, 1, -1 do
      local d_log_y_preds_t = self.criterions[t]:backward(log_y_preds[t], y[{ t, {} }])
      --[[
      if t == T then
        assert(d_hs[t] == nil)
        d_hs[t] = self.softmaxs[t]:backward(hs[t], d_log_y_preds_t)
      else
        d_hs[t]:add( self.softmaxs[t]:backward(hs[t], d_log_y_preds_t) )
      end
      --]]
      d_hs[t]:add( self.softmaxs[t]:backward(hs[t], d_log_y_preds_t) )
      d_embeds[t], d_cs[t-1], d_hs[t-1] = unpack(self.lstms[t]:backward(
        {embeds[t], cs[t-1], hs[t-1]}, 
        {d_cs[t], d_hs[t]})
      )
      self.embs[t]:backward(x[{ t, {} }], d_embeds[t])
    end
    
    --[[
    self.h0s:copy(hs[T])
    self.c0s:copy(cs[T])
    --]]

    -- clip the gradients
    self.grads:clamp(-5, 5)

    return loss, self.grads
  end

  local _, loss_ = optim.adagrad(feval, self.params, sgd_param)
  return loss_[1]
end

