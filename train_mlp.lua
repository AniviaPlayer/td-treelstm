
require '.'
require 'MLP'
require 'hdf5'

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('====== MLP v 1.0 ======')
  cmd:text()
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--useGPU', false, 'use gpu')
  cmd:option('--snhids', '400,300,300,2', 'string hidden sizes for each layer')
  cmd:option('--activ', 'tanh', 'options: tanh, relu')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  cmd:option('--maxEpoch', 10, 'max number of epochs')
  cmd:option('--dataset', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/eot.penn_wsj.conllx.sort.h5', 
    'dataset')
  cmd:option('--ftype', '|x|oe|', '')
  cmd:option('--ytype', 1, '')
  cmd:option('--batchSize', 256, '')
  cmd:option('--lr', 0.01, '')
  cmd:option('--optimMethod', 'AdaGrad', 'options: SGD, AdaGrad')
  cmd:option('--save', 'model.t7', 'save path')
  
  return cmd:parse(arg)
end

local EPOCH_INFO = ''

local DataIter = {}
function DataIter.getNExamples(dataPath, label)
  local h5in = hdf5.open(dataPath, 'r')
  local x_data = h5in:read(string.format('/%s/x', label))
  local N = x_data:dataspaceSize()[1]
  
  return N
end

-- ftype: x | x, e | x, oe | x, e, oe
function DataIter.createBatch(dataPath, label, ftype, ytype, batchSize)
  local h5in = hdf5.open(dataPath, 'r')
  local x_data = h5in:read(string.format('/%s/x', label))
  local e_data = h5in:read(string.format('/%s/e', label))
  local oe_data = h5in:read(string.format('/%s/oe', label))
  local y_data = h5in:read(string.format('/%s/y', label))
  local N = x_data:dataspaceSize()[1]
  local x_width = x_data:dataspaceSize()[2]
  local e_width = e_data:dataspaceSize()[2]
  local oe_width = oe_data:dataspaceSize()[2]
  
  -- print('N = ')
  -- print(N)
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      local x = x_data:partial({istart, iend}, {1, x_width})
      local e = e_data:partial({istart, iend}, {1, e_width})
      local oe = oe_data:partial({istart, iend}, {1, oe_width})
      -- print('OK')
      local y = y_data:partial({istart, iend}, {ytype, ytype}):view(-1) + 1
      -- print('OK, too')
      
      local xd = {}
      if ftype:find('|x|') then
        table.insert(xd, x)
      end
      if ftype:find('|e|') then
        table.insert(xd, e)
      end
      if ftype:find('|oe|') then
        table.insert(xd, oe)
      end
      istart = iend + 1
      
      if #xd == 1 then
        return xd[1], y
      else
        local d = 0
        for i = 1, #xd do
          d = d + xd[i]:size(2)
        end
        local x_ = torch.zeros(x:size(1), d)
        d = 0
        for i = 1, #xd do
          x_[{ {}, {d + 1, d + xd[i]:size(2)} }] = xd[i]
          d = d + xd[i]:size(2)
        end
        
        return x_, y
      end
    else
      h5in:close()
    end
  end
end

local function train(mlp, opts)
  local dataIter = DataIter.createBatch(opts.dataset, 'train', 
    opts.ftype, opts.ytype, opts.batchSize)
  
  local dataSize = DataIter.getNExamples(opts.dataset, 'train')
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, y in dataIter do
    loss = mlp:trainBatch(x, y, sgdParam)
    totalLoss = totalLoss + loss * x:size(1)
    totalCnt = totalCnt + x:size(1)
    
    local ratio = totalCnt/dataSize
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

local function valid(mlp, label, opts)
  local dataIter = DataIter.createBatch(opts.dataset, label, 
    opts.ftype, opts.ytype, opts.batchSize)
  
  local cnt = 0
  local correct, total = 0, 0
  for x, y in dataIter do
    local correct_, total_ = mlp:validBatch(x, y)
    correct = correct + correct_
    total = total + total_
    cnt = cnt + 1
    if cnt % 5 == 0 then collectgarbage() end
  end
  
  return correct, total
end

local function main()
  local opts = getOpts()
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  local mlp = MLP(opts)
  opts.sgdParam = {learningRate = opts.lr}
  opts.curLR = opts.lr
  print(opts)
  
  local timer = torch.Timer()
  local bestAcc = 0
  local bestModel = torch.FloatTensor(mlp.params:size())
  for epoch = 1, opts.maxEpoch do
    EPOCH_INFO = string.format('epoch %d', epoch)
    local startTime = timer:time().real
    local trainCost = train(mlp, opts)
    -- local trainCost = 123
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    local validCor, validTot = valid(mlp, 'valid', opts)
    local validAcc = validCor/validTot
    xprint('VALID %d/%d = %f ', validCor, validTot, validAcc)
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s)', opts.curLR, readableTime(endTime - startTime))
    
    if validAcc > bestAcc then
      bestAcc = validAcc
      mlp:getModel(bestModel)
    end
  end
  
  mlp:setModel(bestModel)
  opts.sgdParam = nil
  mlp:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local validCor, validTot = valid(mlp, 'valid', opts)
  local validAcc = validCor/validTot
  xprint('VALID %d/%d = %f, ', validCor, validTot, validAcc)
  local testCor, testTot = valid(mlp, 'test', opts)
  local testAcc = testCor/testTot
  xprint('TEST %d/%d = %f \n', testCor, testTot, testAcc)
end

main()

