classdef Flow_Layer < dagnn.ElementWise
    
    properties
        Ho = 0;
        Wo = 0;
    end
    
	properties (Transient)
        xx;
        yy;
        zero;
	end

  methods
    function outputs = forward(obj, inputs, params)
        [H,W,~,nbatch] = size(inputs{1});
        obj.Ho = H;
        obj.Wo = W;      
        obj.initGrid();      
        sx = reshape(repmat(obj.xx,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
        sy = reshape(repmat(obj.yy,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
        xxyy_sx = inputs{2}(:,:,1,:)+sx;
        xxyy_sy = inputs{2}(:,:,2,:)+sy;   
        xxyy_s = cat(3,xxyy_sx,xxyy_sy);
        xxyy_s = permute(xxyy_s, [3,1,2,4]);
        outputs{1} = vl_nnbilinearsampler(inputs{1},xxyy_s);
    end
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
        [~,~,~,nbatch] = size(inputs{1});
        sx = reshape(repmat(obj.xx,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
        sy = reshape(repmat(obj.yy,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
        xxyy_sx = inputs{2}(:,:,1,:)+sx;
        xxyy_sy = inputs{2}(:,:,2,:)+sy;   
        xxyy_s = cat(3,xxyy_sx,xxyy_sy);
        xxyy_s = permute(xxyy_s, [3,1,2,4]);
        [derInputs{1},dG] = vl_nnbilinearsampler(inputs{1},xxyy_s,derOutputs{1});
        derInputs{2} = gpuArray(zeros(size(inputs{2}),'single'));        
        derInputs2 = permute(dG,[2,3,1,4]);
        derInputs{2}(:,:,1,:) = derInputs2(:,:,1,:);
        derInputs{2}(:,:,2,:) = derInputs2(:,:,2,:);
        derParams = {};
    end

    function obj = Flow_Layer(varargin)
      obj.load(varargin);
    end
    
    function initGrid(obj)
        xi = linspace(-1,1,obj.Ho);
        yi = linspace(-1,1,obj.Wo);
        [y,x] = meshgrid(xi,yi);
        x = x';
        y = y';
        x = gpuArray(single(x(:)));
        y = gpuArray(single(y(:)));
        obj.xx = y;
        obj.yy = x;
    end    
    
  end
  
end
