
require 'LM_Dataset'

local LM_DatasetGPU, parent = torch.class('LM_DatasetGPU', 'LM_Dataset')

function LM_DatasetGPU:__init(datasetPath, preFetchCount)
  parent.__init(self, datasetPath)
  preFetchCount = preFetchCount or 100
end
