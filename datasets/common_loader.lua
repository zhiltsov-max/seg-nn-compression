local M = {}

function M.loadDataset(name, path, options)
	local load = require ('datasets/' .. name .. '_loader')
    return load(path, options)
end

return M