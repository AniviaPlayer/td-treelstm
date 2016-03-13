
require 'TreeLM_Dataset'

local function main()
  local treelmData = TreeLM_Dataset('/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.dep.100.h5')
  local label, batchSize = 'train', 64
  for x, y in treelmData:createBatch(label, batchSize) do
    print('x = ')
    print(x)
    print('y = ')
    print(y)
  end
end

main()
