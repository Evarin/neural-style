require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'loadcaffe'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_data', 'style.data',
           'Style target, from learn_style.lua')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

function nn.SpatialConvolutionMM:accGradParameters()
   -- nop.  not needed by our net
end

local function main(params)
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 require 'cutorch'
	 require 'cunn'
	 cutorch.setDevice(params.gpu + 1)
      else
	 require 'clnn'
	 require 'cltorch'
	 cltorch.setDevice(params.gpu + 1)
      end
   else
      params.backend = 'nn'
   end

   if params.backend == 'cudnn' then
      require 'cudnn'
      if params.cudnn_autotune then
	 cudnn.benchmark = true
      end
      cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
   end
   
   print('loadcaffe')
   local loadcaffe_backend = params.backend
   if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
   local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 cnn:cuda()
      else
	 cnn:cl()
      end
   end
   
   print('load content')
   local content_image = image.load(params.content_image, 3)
   content_image = image.scale(content_image, params.image_size, 'bilinear')
   local content_image_caffe = preprocess(content_image):float()

   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 content_image_caffe = content_image_caffe:cuda()
      else
	 content_image_caffe = content_image_caffe:cl()
      end
   end
   
   local content_layers = params.content_layers:split(",")

   -- Load style references
   print('load style')
   local pstyle_layers = params.style_layers:split(",")
   local pstyle_ref = torch.load(params.style_data)

   local style_layers = {}
   local style_ref = {}

   for i=1, #pstyle_layers do
      local matching = 0
      for j=1, #pstyle_ref do
	 if pstyle_ref[j].layer == pstyle_layers[i] then
	    matching = j
	    break
	 end
      end
      if matching>0 then
	 table.insert(style_ref, pstyle_ref[matching])
	 table.insert(style_layers, pstyle_layers[i])
      else
	 print('WARNING: ignoring layer '..pstyle_layers[i]..', no matching data in style file')
      end
   end

   pstyle_layers = nil
   pstyle_ref = nil
   collectgarbage()

   print('setting up network')
   -- Set up the network, inserting style and content loss modules
   local content_losses, style_losses = {}, {}
   local next_content_idx, next_style_idx = 1, 1
   local net = nn.Sequential()
   if params.tv_weight > 0 then
      local tv_mod = nn.TVLoss(params.tv_weight):float()
      if params.gpu >= 0 then
	 if params.backend ~= 'clnn' then
	    tv_mod:cuda()
	 else
	    tv_mod:cl()
	 end
      end
      net:add(tv_mod)
   end
   for i = 1, #cnn do
      if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
	 local layer = cnn:get(i)
	 local name = layer.name
	 local layer_type = torch.type(layer)
	 local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
	 if is_pooling and params.pooling == 'avg' then
	    assert(layer.padW == 0 and layer.padH == 0)
	    local kW, kH = layer.kW, layer.kH
	    local dW, dH = layer.dW, layer.dH
	    local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  avg_pool_layer:cuda()
	       else
		  avg_pool_layer:cl()
	       end
	    end
	    local msg = 'Replacing max pooling at layer %d with average pooling'
	    print(string.format(msg, i))
	    net:add(avg_pool_layer)
	 else
	    net:add(layer)
	 end
	 if name == content_layers[next_content_idx] then
	    print("Setting up content layer", i, ":", layer.name)
	    local target = net:forward(content_image_caffe):clone()
	    local norm = params.normalize_gradients
	    local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  loss_module:cuda()
	       else
		  loss_module:cl()
	       end
	    end
	    net:add(loss_module)
	    table.insert(content_losses, loss_module)
	    next_content_idx = next_content_idx + 1
	 end
	 if name == style_layers[next_style_idx] then
	    print("Setting up style layer  ", i, ":", layer.name)
	    local target = style_ref[next_style_idx]
	    local transform, reference = preprocessStyle(target.v, target.e, target.ref)
	    local norm = params.normalize_gradients
	    local loss_module = nn.StyleLoss(params.style_weight, transform, reference, norm):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  loss_module:cuda()
	       else
		  loss_module:cl()
	       end
	    end
	    net:add(loss_module)
	    table.insert(style_losses, loss_module)
	    next_style_idx = next_style_idx + 1
	 end
      end
   end

   -- We don't need the base CNN anymore, so clean it up to save memory.
   cnn = nil
   for i=1,#net.modules do
      local module = net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
	 -- remote these, not used, but uses gpu memory
	 module.gradWeight = nil
	 module.gradBias = nil
      end
   end
   collectgarbage()
   
   -- Initialize the image
   if params.seed >= 0 then
      torch.manualSeed(params.seed)
   end
   local img = nil
   if params.init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
   elseif params.init == 'image' then
      img = content_image_caffe:clone():float()
   else
      error('Invalid init type')
   end
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 img = img:cuda()
      else
	 img = img:cl()
      end
   end
   
   -- Run it through the network once to get the proper size for the gradient
   -- All the gradients will come from the extra loss modules, so we just pass
   -- zeros into the top of the net on the backward pass.
   local y = net:forward(img)
   local dy = img.new(#y):zero()

   -- Declaring this here lets us access it in maybe_print
   local optim_state = nil
   if params.optimizer == 'lbfgs' then
      optim_state = {
	 maxIter = params.num_iterations,
	 verbose=true,
      }
   elseif params.optimizer == 'adam' then
      optim_state = {
	 learningRate = params.learning_rate,
      }
   else
      error(string.format('Unrecognized optimizer "%s"', params.optimizer))
   end

   local function maybe_print(t, loss)
      local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
      if verbose then
	 print(string.format('Iteration %d / %d', t, params.num_iterations))
	 for i, loss_module in ipairs(content_losses) do
	    print(string.format('  Content %d loss: %f', i, loss_module.loss))
	 end
	 for i, loss_module in ipairs(style_losses) do
	    print(string.format('  Style %d loss: %f', i, loss_module.loss))
	 end
	 print(string.format('  Total loss: %f', loss))
      end
   end

   local function maybe_save(t)
      local should_save = params.save_iter > 0 and t % params.save_iter == 0
      should_save = should_save or t == params.num_iterations
      if should_save then
	 local disp = deprocess(img:double())
	 disp = image.minmax{tensor=disp, min=0, max=1}
	 local filename = build_filename(params.output_image, t)
	 if t == params.num_iterations then
	    filename = params.output_image
	 end
	 image.save(filename, disp)
      end
   end

   -- Function to evaluate loss and gradient. We run the net forward and
   -- backward to get the gradient, and sum up losses from the loss modules.
   -- optim.lbfgs internally handles iteration and calls this fucntion many
   -- times, so we manually count the number of iterations to handle printing
   -- and saving intermediate results.
   local num_calls = 0
   local function feval(x)
      num_calls = num_calls + 1
      net:forward(x)
      local grad = net:backward(x, dy)
      local loss = 0
      for _, mod in ipairs(content_losses) do
	 loss = loss + mod.loss
      end
      for _, mod in ipairs(style_losses) do
	 loss = loss + mod.loss
      end
      maybe_print(num_calls, loss)
      maybe_save(num_calls)

      collectgarbage()
      -- optim.lbfgs expects a vector for gradients
      return loss, grad:view(grad:nElement())
   end

   -- Run optimization.
   if params.optimizer == 'lbfgs' then
      print('Running optimization with L-BFGS')
      local x, losses = optim.lbfgs(feval, img, optim_state)
   elseif params.optimizer == 'adam' then
      print('Running optimization with ADAM')
      for t = 1, params.num_iterations do
	 local x, losses = optim.adam(feval, img, optim_state)
      end
   end
end


function build_filename(output_image, iteration)
   local ext = paths.extname(output_image)
   local basename = paths.basename(output_image, ext)
   return string.format('%s_%d.%s', basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
   local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
   local perm = torch.LongTensor{3, 2, 1}
   img = img:index(1, perm):mul(256.0)
   mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
   img:add(-1, mean_pixel)
   return img
end


-- Undo the above preprocessing.
function deprocess(img)
   local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
   mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
   img = img + mean_pixel
   local perm = torch.LongTensor{3, 2, 1}
   img = img:index(1, perm):div(256.0)
   return img
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
   parent.__init(self)
   self.strength = strength
   self.target = target
   self.normalize = normalize or false
   self.loss = 0
   self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
   if input:nElement() == self.target:nElement() then
      self.loss = self.crit:forward(input, self.target) * self.strength
   else
      print('WARNING: Skipping content loss')
   end
   self.output = input
   return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
   if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
   end
   if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
   end
   self.gradInput:mul(self.strength)
   self.gradInput:add(gradOutput)
   return self.gradInput
end


-- process the Style Data
function preprocessStyle(v, e, ref)
   local ev = torch.diag(torch.pow(e + 1.0, -1))
   local transf = v * ev
   local tref = ref * ev
   return transf, tref
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
   local net = nn.Sequential()
   net:add(nn.View(-1):setNumInputDims(2))
   local concat = nn.ConcatTable()
   concat:add(nn.Identity())
   concat:add(nn.Identity())
   net:add(concat)
   net:add(nn.MM(false, true))
   return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, transform, reference, normalize)
   parent.__init(self)
   self.normalize = normalize or false
   self.strength = strength
   self.transform = transform
   self.target = reference:t()
   self.loss = 0
   
   self.gram = GramMatrix()
   self.G = nil
   self.style = nil
   self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
   self.G = self.gram:forward(input)
   self.G:div(input:nElement())
   self.style = torch.mm(self.transform:t(), torch.reshape(self.G, self.G:nElement(), 1))
   self.loss = self.crit:forward(self.style, self.target)
   self.loss = self.loss * self.strength
   self.output = input
   return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
   local dS = self.crit:backward(self.style, self.target)
   dS:div(input:nElement())
   local dG = torch.mm(self.transform, dS)
   self.gradInput = self.gram:backward(input, dG:reshape(dG, self.G:size()))
   if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
   end
   self.gradInput:mul(self.strength)
   self.gradInput:add(gradOutput)
   return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
   parent.__init(self)
   self.strength = strength
   self.x_diff = torch.Tensor()
   self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
   self.output = input
   return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local C, H, W = input:size(1), input:size(2), input:size(3)
   self.x_diff:resize(3, H - 1, W - 1)
   self.y_diff:resize(3, H - 1, W - 1)
   self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
   self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
   self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
   self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
   self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
   self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
   self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
   self.gradInput:mul(self.strength)
   self.gradInput:add(gradOutput)
   return self.gradInput
end


local params = cmd:parse(arg)
main(params)
