local checkpoints = {}

local function deepCopy(tbl)
    -- creates a copy of a network with new modules and the same tensors
    local copy = {}
    for k, v in pairs(tbl) do
        if type(v) == 'table' then
            copy[k] = deepCopy(v)
        else
            copy[k] = v
        end
    end
    if torch.typename(tbl) then
        torch.setmetatable(copy, torch.typename(tbl))
    end
    return copy
end

function checkpoints.load_model_parameters(model_path)
    if not paths.filep(model_path) then
        error('Failed to load model from: ' ..
            'file \'' .. model_path .. '\' is not found.')
    end
    return torch.load(model_path)
end

function checkpoints.save_model_parameters(model, path)
    local model_parameters, _ = model:parameters()
    torch.save(path, model_parameters)
end

function checkpoints.load_solver_state(solver_path)
    if not paths.filep(solver_path) then
        error('Failed to load solver from: ' ..
            'file \'' .. solver_path .. '\' is not found.')
    end
    return torch.load(solver_path)
end

function checkpoints.copy_model_parameters(source_parameters, target_parameters)
    for i = 1, math.min(#source_parameters, #target_parameters) do
        local source_parameter = source_parameters[i]
        local target_parameter = target_parameters[i]

        if (target_parameter:isSameSizeAs(source_parameter)) then
            target_parameter:copy(source_parameter)
            print(string.format("Successfully copied layer #%d of shape %s",
                i,
                table.concat(torch.totable(target_parameter:size()), 'x')
            ))
        else
            print(string.format(
                "Failed to copy layer #%d: source shape %s, target shape %s",
                i,
                table.concat(torch.totable(source_parameter:size()), 'x'),
                table.concat(torch.totable(target_parameter:size()), 'x')
            ))
        end
    end
end

function checkpoints.restore_model_state(source_state, model)
    local source_parameters = source_state.parameters
    local source_gradients = source_state.gradients
    local target_parameters, target_gradients = model:parameters() -- may allocate new storage

    assert((#source_parameters == #target_parameters) and 
        (#source_gradients == #target_gradients) and 
        (#source_parameters == #target_gradients),
        "Expected equal sizes.")
    for i = 1, #source_parameters do
        target_parameters[i]:copy(source_parameters[i])
        target_gradients[i]:copy(source_gradients[i])
    end
end

function checkpoints.load(checkpoint_path)
    local checkpoint_info = torch.load(checkpoint_info, 'ascii')
    local checkpoint_root_path = paths.dirname(checkpoint_path)
    local model_path = paths.concat(checkpoint_root_path, checkpoint_info.model_file)
    local solver_path = paths.concat(checkpoint_root_path, checkpoint_info.solver_file)

    local model_parameters = checkpoints.load_model_parameters(model_path)
    local solver_state = checkpoints.load_solver_state(solver_path)

    local model = {
        parameters = model_parameters,
        gradients = solver_state.gradients
    }
    local solver = solver_state.solver_state

    return model, solver
end

function checkpoints.save(model, solver, path)
    -- don't save the DataParallelTable for easier loading on other machines
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    local solver_state = solver:get_state()
    local epoch = solver_state.epoch
    local model_file = 'model_' .. epoch .. '.t7'
    local solver_file = 'solver_' .. epoch .. '.t7'
    local checkpoint_file = 'snapshot_' .. epoch .. '.t7'

    local model_parameters, model_gradients = model:parameters()

    local checkpoint_info = {
        model_file = model_file,
        solver_file = solver_file
    }

    local saved_solver_state = {
        solver_state = solver_state,
        gradients = model_gradients
    }

    torch.save(paths.concat(path, model_file), model_parameters)
    torch.save(paths.concat(path, solver_file), saved_solver_state)
    torch.save(paths.concat(path, checkpoint_file), checkpoint_info, 'ascii')
end

return checkpoints