function loadDataset(name, path, options)
    if name == 'camvid' then
        local load = require 'datasets/camvid_loader'
        return load(path, options)
    else
        error("Dataset loader for '" .. opt.dataset .. "' is not found.")
    end
end