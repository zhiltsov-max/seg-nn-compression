local checkpoints = require 'checkpoints'

local models = {}

function models.init(options)
    local model, cost = paths.dofile(options.model)(options)
    if (options.transferFrom ~= 'none') then
        local source_model_parameters = models.load(options.transferFrom)
        local weights, _ = model:parameters()
        print('Transferring weights from \'' .. options.transferFrom .. '\'...')
        checkpoints.copy_model_parameters(source_model_parameters, weights)
    end

    cudnn.convert(model, cudnn)

   -- optnet is an general library for reducing memory usage in neural networks
   if (options.optnet == true) then
        local optnet = require 'optnet'
        print('Trying to optimize memory usage with optnet...')

        local sampleInput = torch.zeros(options.batchSize, 3, options.imHeight, options.imWidth):type(options.tensorType):cuda()

        local function checkMemory(model, params)
            collectgarbage()
            model:forward(sampleInput)
            mem1 = optnet.countUsedMemory(model)

            optnet.optimizeMemory(model, sampleInput, params)

            collectgarbage()
            model:forward(sampleInput)
            mem2 = optnet.countUsedMemory(model)

            print('Memory usage:')
            print('Before optimization: ' .. mem1.total_size/1024/1024 .. ' MB')
            print(mem1)
            print('After optimization : ' .. mem2.total_size/1024/1024 .. ' MB')
            print(mem2)
        end

        print("Checking memory usage in training mode...")
        checkMemory(model, {inplace = false, mode = 'training'})
   end

   return model, cost
end

function models.restore_model_state(state, model)
    return checkpoints.restore_model_state(state, model)
end

function models.load(model_path)
    return checkpoints.load_model_parameters(model_path)
end

function models.save(model, path)
    checkpoints.save_model_parameters(model, path)
end


return models