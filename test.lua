require 'xlua'
require 'optim'
require 'image'
require 'paths'
require 'cutorch'


local function rfind(str, pattern)
    local pos = str:find(pattern)
    local cur_pos = pos
    while (cur_pos ~= nil) do
        pos = cur_pos
        cur_pos = str:find(pattern, cur_pos + 1)
    end
    return pos
end

local function get_file_name(path)
    local begin_idx = (rfind(path, "/") or 0) + 1
    local end_idx = (path:find(".", begin_idx, true) or 0) - 1
    return path:sub(begin_idx, end_idx)
end

local function test_subset(model, subset, validate, save_dir, options)
    local confusion_matrix = nil
    if (validate == true) then
        confusion_matrix = optim.ConfusionMatrix(subset.class_count, subset.classes)
        confusion_matrix:zero()
    end

    model:evaluate()

    local test_data = subset.data
    local test_data_size = #test_data
    print("Running for " .. test_data_size .. " images...")

    os.execute('mkdir -p ' .. save_dir)
    local save_dir_raw = paths.concat(save_dir, 'raw')
    os.execute('mkdir -p ' .. save_dir_raw)
    local save_dir_painted = paths.concat(save_dir, 'painted')
    os.execute('mkdir -p ' .. save_dir_painted)

    local test_list = subset.list

    -- Preallocation for inputs
    local image_size = {options.imHeight, options.imWidth}
    local input = torch.Tensor(1,
        options.inputChannelsCount, options.imHeight, options.imWidth
    ):type(options.tensorType)

    local total_time = 0

    for i = 1, test_data_size do
        if i % math.max(1, math.floor(test_data_size / 10)) then
            io.write("\r", "Inference: ", i, "/", test_data_size, "\r")
        end
        
        input:zero()
        
        local input_image = test_data[i][1]
        input:copy(input_image)

        cutorch.synchronize()
        local time = sys.clock()
        local output = model:forward(input)
        cutorch.synchronize()
        time = sys.clock() - time
        total_time = total_time + time

        local prediction = output[1]

        local max_values, max_indices = torch.max(prediction, 1)
        max_indices = max_indices[1][{{1, image_size[1]}, {1, image_size[2]}}]
        
        local image_name = get_file_name(test_list[i][1])
        
        if (validate == true) then
            local target = test_data[i][2]
            local mask = target:lt(255)
            confusion_matrix:batchAdd(max_indices[mask]:view(-1), target[mask]:view(-1))

            -- local target_image_painted = subset:paint_segmentation(target:float()):mul(1.0 / 255.0)
            -- local filepath_painted = paths.concat(save_dir, image_name .. ".png")
            -- image.save(filepath_painted, target_image_painted)
        end
        
        local output_image_raw = max_indices:float():mul(1.0 / 255.0)
        local filepath_raw = paths.concat(save_dir_raw, image_name .. ".png")
        image.save(filepath_raw, output_image_raw)
        
        local output_image_painted = subset:paint_segmentation(max_indices:float()):mul(1.0 / 255.0)
        local filepath_painted = paths.concat(save_dir_painted, image_name .. ".png")
        image.save(filepath_painted, output_image_painted)
    end
    
    print(string.format("Inference time: %.3fs", total_time))
    print(string.format("Average time per image: %.3fs", total_time / test_data_size))

    if (validate == true) then
        print(confusion_matrix)
    end

    collectgarbage()
end

local function test(model, dataset, subset_name, validate, save_dir, options)
    local mt = { __index = dataset }
    setmetatable(mt, dataset)
    local subset = {}
    setmetatable(subset, mt)
    if (subset_name == 'train') then
        subset.list = dataset.train_list
        subset.data = dataset.train_data
    elseif (subset_name == 'val') then
        subset.list = dataset.val_list
        subset.data = dataset.val_data
    elseif (subset_name == 'test') then
        subset.list = dataset.test_list
        subset.data = dataset.test_data
    else
        error('Unknown subset \'' .. subset_name .. '\'')
    end
    test_subset(model, subset, validate, save_dir, options)
end

return test
