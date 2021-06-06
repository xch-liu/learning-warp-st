classdef Transformer_Flow_Layer < dagnn.ElementWise
    
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
        [H,W,Z,nbatch] = size(inputs{1});
        obj.Ho = H;
        obj.Wo = W;      
        obj.initGrid();      
        
        outputs{1} = gpuArray(zeros(H*3,W*3,Z,nbatch,'single'));
        for i = 1:3
            for j = 1:3
                sx = reshape(repmat(obj.xx,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
                sy = reshape(repmat(obj.yy,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
                xxyy_sx = inputs{2}(:,:,1,:)*(2/(obj.Ho-1))*(i-2)+inputs{2}(:,:,2,:)*(2/(obj.Wo-1))*(j-2)+inputs{2}(:,:,3,:)+sx;
                xxyy_sy = inputs{2}(:,:,4,:)*(2/(obj.Ho-1))*(i-2)+inputs{2}(:,:,5,:)*(2/(obj.Wo-1))*(j-2)+inputs{2}(:,:,6,:)+sy; 
                xxyy_s = cat(3,xxyy_sx,xxyy_sy);
                xxyy_s = permute(xxyy_s, [3,1,2,4]);
                outputs{1}(i:3:end,j:3:end,:,:) = vl_nnbilinearsampler(inputs{1},xxyy_s);
            end
        end
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
        [~,~,~,nbatch] = size(inputs{1});
        derInputs{1} = gpuArray(zeros(size(inputs{1}),'single'));
        derInputs{2} = gpuArray(zeros(size(inputs{2}),'single'));
        for i = 1:3
            for j = 1:3
                sx = reshape(repmat(obj.xx,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
                sy = reshape(repmat(obj.yy,1,nbatch),obj.Ho,obj.Wo,1,nbatch);
                xxyy_sx = inputs{2}(:,:,1,:)*(2/(obj.Ho-1))*(i-2)+inputs{2}(:,:,2,:)*(2/(obj.Wo-1))*(j-2)+inputs{2}(:,:,3,:)+sx;
                xxyy_sy = inputs{2}(:,:,4,:)*(2/(obj.Ho-1))*(i-2)+inputs{2}(:,:,5,:)*(2/(obj.Wo-1))*(j-2)+inputs{2}(:,:,6,:)+sy; 
                xxyy_s = cat(3,xxyy_sx,xxyy_sy);
                xxyy_s = permute(xxyy_s, [3,1,2,4]);
                [derInputs1,dG] = vl_nnbilinearsampler(inputs{1},xxyy_s,derOutputs{1}(i:3:end,j:3:end,:,:));
                derInputs{1} = derInputs{1}+derInputs1;
                derInputs2 = permute(dG,[2,3,1,4]);
%                 derInputs{2}(:,:,1,:) = derInputs{2}(:,:,1,:)+derInputs2(:,:,1,:)*(2/(obj.Ho-1))*(i-2);
%                 derInputs{2}(:,:,2,:) = derInputs{2}(:,:,2,:)+derInputs2(:,:,1,:)*(2/(obj.Wo-1))*(j-2);
                derInputs{2}(:,:,3,:) = derInputs{2}(:,:,3,:)+derInputs2(:,:,1,:);
%                 derInputs{2}(:,:,4,:) = derInputs{2}(:,:,4,:)+derInputs2(:,:,2,:)*(2/(obj.Ho-1))*(i-2);
%                 derInputs{2}(:,:,5,:) = derInputs{2}(:,:,5,:)+derInputs2(:,:,2,:)*(2/(obj.Wo-1))*(j-2);
                derInputs{2}(:,:,6,:) = derInputs{2}(:,:,6,:)+derInputs2(:,:,2,:);
            end
        end
        derParams = {};
    end

    function obj = Transformer_Flow_Layer(varargin)
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
