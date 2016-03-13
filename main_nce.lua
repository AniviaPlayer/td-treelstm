
require '.'
require 'shortcut'
require 'TreeLSTMLM'
require 'TreeLM_Dataset'

require 'TreeLSTMNCELM'
require 'TreeLM_NCE_Dataset'

local model_utils = require 'model_utils'
local EPOCH_INFO = ''

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== Tree LSTM NCE Language Model ======')
  cmd:text('version 2.2 add word embedding support')
  cmd:text()
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--model', 'TreeLSTM', 'model options: TreeLSTM, TreeLSTMNCE')
  cmd:option('--dataset', '', 'dataset path')
  cmd:option('--maxEpoch', 100, 'maximum number of epochs')
  cmd:option('--batchSize', 64, '')
  cmd:option('--validBatchSize', 16, '')
  cmd:option('--nin', 50, 'word embedding size')
  cmd:option('--nhid', 100, 'hidden unit size')
  cmd:option('--nlayers', 1, 'number of hidden layers')
  cmd:option('--wordEmbedding', '', 'path for the word embedding file')
  cmd:option('--lr', 0.1, 'learning rate')
  cmd:option('--lrDiv', 0, 'learning rate decay when there is no significant improvement. 0 means turn off')
  cmd:option('--minImprovement', 1.0001, 'if improvement on log likelihood is smaller then patient --')
  cmd:option('--optimMethod', 'AdaGrad', 'optimization algorithm')
  cmd:option('--gradClip', 5, '> 0 means to do Pascanu et al.\'s grad norm rescale http://arxiv.org/pdf/1502.04623.pdf; < 0 means to truncate the gradient larger than gradClip; 0 means turn off gradient clip')
  cmd:option('--initRange', 0.1, 'init range')
  cmd:option('--initHidVal', 0.01, 'init values for hidden states')
  cmd:option('--seqLen', 151, 'maximum seqence length')
  cmd:option('--useGPU', false, 'use GPU')
  cmd:option('--patience', 2, 'stop training if no lower valid PPL is observed in [patience] consecutive epoch(s)')
  cmd:option('--save', 'model.t7', 'save model path')
  
  cmd:text()
  cmd:text('Options for NCE')
  cmd:option('--nneg', 20, 'number of negative samples')
  cmd:option('--power', 0.75, 'for power for unigram frequency')
  cmd:option('--lnZ', 9.5, 'default normalization term')
  cmd:option('--learnZ', false, 'learn the normalization constant Z')
  cmd:option('--normalizeUNK', false, 'if normalize UNK or not')
  
  cmd:text()
  cmd:text('Options for long jobs')
  cmd:option('--savePerEpoch', false, 'save model every epoch')
  cmd:option('--saveBeforeLrDiv', false, 'save model before lr div')
  
  cmd:text()
  cmd:text('Options for regularization')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  
  return cmd:parse(arg)
end

local function train(rnn, lmdata, opts)
  local dataIter
  if opts.model:find('NCE') then
    dataIter = lmdata:createBatch('train', opts.batchSize, true)
  else
    dataIter = lmdata:createBatch('train', opts.batchSize)
  end
  
  local dataSize, curDataSize = lmdata:getTrainSize(), 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, y, y_neg, y_prob, y_neg_prob, mask in dataIter do
    local loss
    if y_neg then
      loss = rnn:trainBatch(x, y, y_neg, y_prob, y_neg_prob, mask, sgdParam)
    else
      loss = rnn:trainBatch(x, y, sgdParam)
    end
    
    local nll = loss * x:size(2) / (y:ne(0):sum())
    if mask then
      nll = loss * x:size(2) / (mask:sum())
    else
      nll = loss * x:size(2) / (y:ne(0):sum())
    end
    
    totalLoss = totalLoss + loss * x:size(2)
    if mask then
      totalCnt = totalCnt + mask:sum()
    else
      totalCnt = totalCnt + y:ne(0):sum()
    end
    
    curDataSize = curDataSize + x:size(2)
    local ratio = curDataSize/dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\r%s %.3f %.4f (%s) / %.2f wps ... ', EPOCH_INFO, ratio, totalLoss/totalCnt, readableTime(timer:time().real), wps )
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  return totalLoss / totalCnt
end

local function valid(rnn, lmdata, opts, splitLabel)
  rnn:disableDropout()
  
  local dataIter = lmdata:createBatch(splitLabel, opts.validBatchSize)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  for x, y in dataIter do
    local loss = rnn:validBatch(x, y)
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y:ne(0):sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  rnn:enableDropout()
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  return {entropy = entropy, ppl = ppl}
end

local function verifyModel(modelPath)
  xprintln('\n==verify trained model==')
  local optsPath = modelPath:sub(1, -4) .. '.state.t7'
  local opts = torch.load(optsPath)
  xprintln('load state from %s done!', optsPath)
  
  print(opts)
  local lmdata = nil
  if opts.model == 'TreeLSTM' then
    lmdata = TreeLM_Dataset(opts.dataset)
  elseif opts.model == 'TreeLSTMNCE' then
    lmdata = TreeLM_NCE_Dataset(opts.dataset, opts.nneg, opts.power, opts.normalizeUNK)
  end
  -- local lmdata = TreeLM_Dataset(opts.dataset)
  
  local rnn
  if opts.model == 'TreeLSTM' then
    rnn = TreeLSTMLM(opts)
  elseif opts.model == 'TreeLSTMNCE' then
    rnn = TreeLSTMNCELM(opts)
  end
  
  -- local rnn = TreeLSTMLM(opts)
  xprintln( 'load model from %s', opts.save )
  rnn:load(opts.save)
  xprintln( 'load model from %s done!', opts.save )
  
  xprintln('\n')
  local validRval = valid(rnn, lmdata, opts, 'valid')
  xprint('VALID %f ', validRval.ppl)
  local testRval = valid(rnn, lmdata, opts, 'test')
  xprintln('TEST %f ', testRval.ppl)
end

local function initOpts(opts)
  -- for different models
  local nceParams = {'nneg', 'power', 'normalizeUNK', 'learnZ', 'lnZ'}
  if opts.model == 'TreeLSTM' then
    -- delete nce params
    for _, nceparam in ipairs(nceParams) do
      opts[nceparam] = nil
    end
  end
  
  -- for different optimization algorithms
  local optimMethods = {'AdaGrad', 'Adam', 'AdaDelta', 'SGD'}
  if not table.contains(optimMethods, opts.optimMethod) then
    error('invalid optimization problem ' .. opts.optimMethod)
  end
  
  opts.curLR = opts.lr
  opts.minLR = 1e-7
  opts.sgdParam = {learningRate = opts.lr}
  if opts.optimMethod == 'AdaDelta' then
    opts.rho = 0.95
    opts.eps = 1e-6
    opts.sgdParam.rho = opts.rho
    opts.sgdParam.eps = opts.eps
  elseif opts.optimMethod == 'SGD' then
    if opts.lrDiv <= 1 then
      opts.lrDiv = 2
    end
  end
  
end

local function main()
  local opts = getOpts()
  print('version 2.2 add word embedding support')
  
  initOpts(opts)
  
  local lmdata = nil
  if opts.model == 'TreeLSTM' then
    lmdata = TreeLM_Dataset(opts.dataset)
  elseif opts.model == 'TreeLSTMNCE' then
    lmdata = TreeLM_NCE_Dataset(opts.dataset, opts.nneg, opts.power, opts.normalizeUNK)
  end
  opts.nvocab = lmdata:getVocabSize()
  
  print(opts)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  local rnn = nil
  if opts.model == 'TreeLSTM' then
    rnn = TreeLSTMLM(opts)
  elseif opts.model == 'TreeLSTMNCE' then
    rnn = TreeLSTMNCELM(opts)
  end
  
  local bestValid = {ppl = 1e309, entropy = 1e309}
  local lastValid = {ppl = 1e309, entropy = 1e309}
  local bestModel = torch.FloatTensor(rnn.params:size())
  local patience = opts.patience
  local divLR = false
  local timer = torch.Timer()
  local epochNo = 0
  for epoch = 1, opts.maxEpoch do
    epochNo = epochNo + 1
    EPOCH_INFO = string.format('epoch %d', epoch)
    local startTime = timer:time().real
    local trainCost = train(rnn, lmdata, opts)
    -- print('training ignored!!!')
    -- local trainCost = 123
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    local validRval = valid(rnn, lmdata, opts, 'valid')
    xprint('VALID %f ', validRval.ppl)
    --[[
    local testRval = valid(rnn, lmdata, opts, 'test')
    xprint('TEST %f ', testRval.ppl)
    --]]
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s) p = %d', opts.curLR, readableTime(endTime - startTime), patience)
    
    if validRval.ppl < bestValid.ppl then
      bestValid.ppl = validRval.ppl
      bestValid.entropy = validRval.entropy
      bestValid.epoch = epoch
      rnn:getModel(bestModel)
      -- for non SGD algorithm, we will reset the patience
      -- if opts.optimMethod ~= 'SGD' then
      if opts.lrDiv <= 1 then
        patience = opts.patience
      end
    else
      -- non SGD algorithm decrease patience
      if opts.lrDiv <= 1 then
      -- if opts.optimMethod ~= 'SGD' then
        patience = patience - 1
        if patience == 0 then
          xprintln('No improvement on PPL for %d epoch(s). Training finished!', opts.patience)
          break
        end
      else
        -- SGD with learning rate decay
        rnn:setModel(bestModel)
      end
      
    end -- if validRval.ppl < bestValid.ppl
    
    if opts.savePerEpoch then
      local tmpPath = opts.save:sub(1, -4) .. '.tmp.t7'
      rnn:save(tmpPath, true)
    end
    
    if opts.saveBeforeLrDiv then
      if opts.optimMethod == 'SGD' and opts.curLR == opts.lr then
        local tmpPath = opts.save:sub(1, -4) .. '.blrd.t7'
        rnn:save(tmpPath, true)
      end
    end
    
    -- control the learning rate decay
    -- if opts.optimMethod == 'SGD' then
    if opts.lrDiv > 1 then
      if epoch >= 10 and patience > 1 then
        patience = 1
      end
      
      if validRval.entropy * opts.minImprovement > lastValid.entropy then
        if not divLR then  -- patience == 1
          patience = patience - 1
          if patience < 1 then divLR = true end
        else
          xprintln('no significant improvement! cur ppl %f, best ppl %f', validRval.ppl, bestValid.ppl)
          break
        end
      end
      
      if divLR then
        opts.curLR = opts.curLR / opts.lrDiv
        opts.sgdParam.learningRate = opts.curLR
      end
      
      if opts.curLR < opts.minLR then
        xprintln('min lr is met! cur lr %e min lr %e', opts.curLR, opts.minLR)
        break
      end
      lastValid.ppl = validRval.ppl
      lastValid.entropy = validRval.entropy
    end
  end
  
  if epochNo > opts.maxEpoch then
    xprintln('Max number of epoch is met. Training finished!')
  end
  
  lmdata:close()
  
  rnn:setModel(bestModel)
  opts.sgdParam = nil
  rnn:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  -- verifyModel(opts.save)
end

-- here is the entry
main()

