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


   -- optnet is an general library for reducing memory usage in neural networks
   if (options.optnet == true) then
        local optnet = require 'optnet'
        print('Trying to optimize memory usage with optnet...')

        local sampleInput = torch.zeros(1, 3, options.imHeight, options.imWidth):type(options.tensorType)

        collectgarbage()
        model:forward(sampleInput)
        mem1 = optnet.countUsedMemory(model)

        optnet.optimizeMemory(model, sampleInput, {inplace = true, reuseBuffers = true, mode = 'training'})

        collectgarbage()
        model:forward(sampleInput)
        mem2 = optnet.countUsedMemory(model)

        print('Memory usage:')
        print('Before optimization : '.. mem1.total_size/1024/1024 .. ' MB')
        print('After optimization  : '.. mem2.total_size/1024/1024 .. ' MB')

        if (mem1.total_size < mem2.total_size) then
            print('Unrolling optimizations')
            
            optnet.removeOptimization(model)
            
            collectgarbage()
            model:forward(sampleInput)
            mem3 = optnet.countUsedMemory(model)
            
            print('After removing optimization: '.. mem3.total_size/1024/1024 .. ' MBytes')
        end
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