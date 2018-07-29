require 'optim'
require 'cunn'

local solver = {}

function solver.init(self, model, cost, dataset, options, previous_state)
    self.state = previous_state or {
        sgd = {
            learningRate = options.learningRate,
            momentum = options.momentum,
	        weightDecay = options.weightDecay
        },
        basicLrDecay = options.lrDecay,
        epoch = 1
    }

    -- 'type()' used instead of 'cuda()' to ensure correct tensor type
    -- 'cuda()' returns CudaTensor module, while there are 
    -- 'cudaHalf()' and others, that convert model to corresponding types.
    self.model = model:type(options.tensorType)
    self.cost = cost:type(options.tensorType)

    self.dataset = dataset
    self.train_data, _ = dataset:get_iterators()
    self.image_size = {height = options.imHeight, width = options.imWidth}

    self.batch_count = math.ceil(#self.dataset.train_data / options.batchSize)
    self.batch_size = options.batchSize

    self.inputs = torch.Tensor(options.batchSize,
        options.inputChannelsCount, options.imHeight, options.imWidth
    ):type(options.tensorType)
    self.targets = torch.Tensor(options.batchSize,
        options.imHeight, options.imWidth
    ):type(options.tensorType)

    -- Normalize LRd to batch count, so learning with different batch sizes
    -- gives comparable learning dynamics
    self.state.sgd.learningRateDecay = self.state.basicLrDecay / self.batch_count

    self.weights, self.dE_dw = model:getParameters() -- should be called after model:cuda()

    return self
end

function solver.get_state(self)
    return self.state
end

function solver.get_epoch(self)
    return self.state.epoch
end

function solver.set_epoch(self, epoch)
    self.state.epoch = epoch
end

function solver.load_batch(self, batch_idx)
    local shuffle = self.shuffle
    local train_data = self.train_data

    self.inputs:zero()
    self.targets:zero()

    local current_batch_size = self.batch_size
    if (batch_idx == self.batch_count) then
        local rest = #self.dataset.train_data % self.batch_size
        if (rest ~= 0) then
            current_batch_size = rest
        end
    end
    
    for i = 1, current_batch_size do
        local dataset_index = shuffle[(batch_idx - 1) * self.batch_size + i]
        local dataset_entry = train_data[dataset_index]
        self.inputs[i]:copy(dataset_entry[1])
        self.targets[i]:copy(dataset_entry[2])
    end

    -- Return only meaningful part of batch in the case of incomplete batch
    return self.inputs [{{1, current_batch_size}}], 
           self.targets[{{1, current_batch_size}}]
end

function solver.shuffle_data(self)
    self.shuffle = torch.randperm(#self.train_data)
end

function solver.run_training_epoch(self)
    local epoch = self.state.epoch
    local model = self.model
    local cost = self.cost
    local weights = self.weights
    local sgd = self.state.sgd
    local dE_dw = self.dE_dw

    self:shuffle_data()

    model:training()

    local epoch_time = 0

    local batch_count = self.batch_count
    for batch_id = 1, batch_count do
        local batch_time = sys.clock()

        local x, y = self:load_batch(batch_id)

        -- create closure to evaluate E(W) and dE/dW
        local eval_E = function(weights)
            -- evaluate function for complete batch
            local f = model:forward(x)
            local loss = cost:forward(f, y)

            -- reset gradients
            dE_dw:zero()
            -- estimate gradients dE_dw (stored in model)
            local dE_df = cost:backward(f, y)
            model:backward(x, dE_df)

            return loss, dE_dw
        end

        local _, loss = optim.sgd(eval_E, weights, sgd)

        batch_time = sys.clock() - batch_time
        epoch_time = epoch_time + batch_time

        io.write(
            "\r",
            string.format(
                "Epoch #%d (%d/%d)" ..
                    ", time %.3fs" ..
                    ", batch time %.3fs" ..
                " | " ..
                    "lr %.2e" ..
                    ", loss %.4f",
                epoch, batch_id, batch_count,
                epoch_time, batch_time,
                sgd.learningRate / (1 + sgd.evalCounter * sgd.learningRateDecay), -- from optim/sgd
                loss[1]
            ),
            "\r")
    end

    self:set_epoch(epoch + 1)

    collectgarbage()

    return model, loss
end

return solver