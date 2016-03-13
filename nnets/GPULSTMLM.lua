
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'Embedding'
require 'MaskedClassNLLCriterion'

require 'basic'

local model_utils = require 'model_utils'

local GPULSTMLM = torch.class('GPULSTMLM', 'BModel')

local function transferData(useGPU, data)
  if useGPU then
    return data:cuda()
  else
    return data
  end
end

function GPULSTMLM:__init(opts)
  self.opts = opts
  self.name = 'GPULSTMLM'
  self:print( 'build LSTMLM ...' )
  -- torch.manualSeed(opts.seed)
  -- build model
  opts.nivocab = opts.nivocab or opts.nvocab
  opts.novocab = opts.novocab or opts.nvocab
  opts.seqlen = opts.seqlen or 10
  self.coreNetwork = self:createNetwork(opts)
  self.params, self.grads = self.coreNetwork:getParameters()
  self.params:uniform(-opts.initRange, opts.initRange)
  print(self.params:size())
  print(self.params[{ {1, 10} }])
  
  self:print( 'Begin to clone model' )
  self.networks = model_utils.clone_many_times(self.coreNetwork, opts.seqLen)
  self:print( 'Clone model done!' )
  
  self:print('init states')
  self:setup(opts)
  self:print('init states done!')
  
  self:print( 'build LSTMLM done!' )
end

function GPULSTMLM:setup(opts)
  self.hidStates = {}   -- including all h_t and c_t
  self.initStates = {}
  self.df_hidStates = {}
  self.df_StatesT = {}
  
  for i = 1, 2*opts.nlayers do
    self.initStates[i] = transferData(opts.useGPU, torch.ones(opts.batchSize, opts.nhid) * opts.initHidVal)
    self.df_StatesT[i] = transferData(opts.useGPU, torch.zeros(opts.batchSize, opts.nhid))
  end
  self.hidStates[0] = self.initStates
  self.err = transferData(opts.useGPU, torch.zeros(opts.seqLen))
end

function GPULSTMLM:createLSTM(x_t, c_tm1, h_tm1, nin, nhid)
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

function GPULSTMLM:createNetwork(opts)
  local x_t = nn.Identity()()
  local y_t = nn.Identity()()
  local s_tm1 = nn.Identity()()
  local in_t = {[0] = nn.LookupTable(opts.nivocab, opts.nin)(x_t)}
  -- local in_t = {[0] = Embedding(opts.nivocab, opts.nin)(x_t)}
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
  
  local h2y = nn.Linear(opts.nhid, opts.novocab)(in_t[opts.nlayers])
  local y_pred = nn.LogSoftMax()(h2y)
  local err = MaskedClassNLLCriterion()({y_pred, y_t})
  
  local model = nn.gModule({x_t, y_t, s_tm1}, {nn.Identity()(s_t), err})
  if opts.useGPU then
    return model:cuda()
  else
    return model
  end
end

function GPULSTMLM:trainBatch(x, y, sgdParam)
  --[[
  x = x:type('torch.DoubleTensor')
  y = y:type('torch.DoubleTensor')
  --]]
  if self.opts.useGPU then
    x = x:cuda()
    y = y:cuda()
  end
  local function feval(params_)
    if self.params ~= params_ then
      self.params:copy(params_)
    end
    self.grads:zero()
    -- forward pass
    local loss = 0
    local T = x:size(1)
    for t = 1, T do
      local s_tm1 = self.hidStates[t - 1]
      self.hidStates[t], self.err[t] = 
        unpack( self.networks[t]:forward({ x[{ t, {} }], y[{ t, {} }], s_tm1 }) )
      loss = loss + self.err[t]
    end
    
    for i = 1, 2*self.opts.nlayers do
      self.df_StatesT[i]:zero()
    end
    self.df_hidStates[T] = self.df_StatesT
    
    for t = T, 1, -1 do
      local s_tm1 = self.hidStates[t - 1]
      local derr = transferData(self.opts.useGPU, torch.ones(1))
      local _, _, df_hidStates_tm1 = unpack(
        self.networks[t]:backward(
          {x[{ t, {} }], y[{ t, {} }], s_tm1},
          {self.df_hidStates[t], derr}
          )
        )
      self.df_hidStates[t-1] = df_hidStates_tm1
      
      if self.opts.useGPU then
        cutorch.synchronize()
      end
    end
    
    -- clip the gradients
    self.grads:clamp(-5, 5)
    
    return loss, self.grads
  end
  
  local _, loss_ = optim.adagrad(feval, self.params, sgdParam)
  return loss_[1]
end

function GPULSTMLM:validBatch(x, y)
  local loss = 0
  local T = x:size(1)
  for t = 1, T do
    local s_tm1 = self.hidStates[t - 1]
    self.hidStates[t], self.err[t] = 
      unpack( self.networks[t]:forward({ x[{ t, {} }], y[{ t, {} }], s_tm1 }) )
    loss = loss + self.err[t]
  end
  
  return loss
end


