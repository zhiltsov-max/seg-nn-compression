local datasets = {}

function datasets.loadDataset(name, path, options)
    if (name == 'camvid') then
        local load = require 'datasets/camvid_loader'
        return load(path, options)
    elseif (name == 'camvid12') then
        local load = require 'datasets/camvid12_loader'
        return load(path, options)
    else
        error("Dataset loader for '" .. opt.dataset .. "' is not found.")
    end
end

return datasets