classdef Constraint_Correlation_Layer < dagnn.ElementWise
    
    properties
        Ho = 0;
        Wo = 0;
        max_disp;
        stride;
    end
    
	properties (Transient)
        xx;
        yy;
	end

  methods
    function outputs = forward(obj, inputs, params)
      [H,W,~,nbatch] = size(inputs{1});
      center_max_disp = (obj.max_disp+1)/2;
      
      obj.Ho = H;
      obj.Wo = W;      
      obj.initGrid();
      
     outputs{1} = zeros(H,W,obj.max_disp*obj.max_disp,nbatch,'single');
      win_size = 0;
      for j = 1:obj.max_disp
          for i = 1:obj.max_disp
              win_size = win_size + 1;
              xxyy_sx = repmat(obj.xx,1,nbatch)+(2/(obj.Ho-1))*(i-center_max_disp)*obj.stride;
              xxyy_sy = repmat(obj.yy,1,nbatch)+(2/(obj.Wo-1))*(j-center_max_disp)*obj.stride;
              xxyy_sx =  reshape(xxyy_sx,obj.Ho,obj.Wo,1,nbatch);
              xxyy_sy =  reshape(xxyy_sy,obj.Ho,obj.Wo,1,nbatch);
              xxyy_s = cat(3,xxyy_sx,xxyy_sy);
              xxyy_s = permute(xxyy_s, [3,1,2,4]);
              inputs2 = vl_nnbilinearsampler(inputs{2},xxyy_s);
              outputs{1}(:,:,win_size,:) = sum(inputs{1}.*inputs2,3);              
          end
      end
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
        [~,~,depth,nbatch] = size(inputs{1});
        center_max_disp = (obj.max_disp+1)/2;   
        derInputs{1} = zeros(size(inputs{1}),'single');
        derInputs{2} = zeros(size(inputs{2}),'single');
        win_size = 0;
        for j = 1:obj.max_disp
            for i = 1:obj.max_disp
                  win_size = win_size + 1;
                  xxyy_sx = repmat(obj.xx,1,nbatch)+(2/(obj.Ho-1))*(i-center_max_disp)*obj.stride; 
                  xxyy_sy = repmat(obj.yy,1,nbatch)+(2/(obj.Wo-1))*(j-center_max_disp)*obj.stride;          
                  xxyy_sx =  reshape(xxyy_sx,obj.Ho,obj.Wo,1,nbatch);
                  xxyy_sy =  reshape(xxyy_sy,obj.Ho,obj.Wo,1,nbatch);
                  xxyy_s = cat(3,xxyy_sx,xxyy_sy);
                  xxyy_s = permute(xxyy_s, [3,1,2,4]);
                  inputs2 = vl_nnbilinearsampler(inputs{2},xxyy_s);
                  derOutputs1 = repmat(derOutputs{1}(:,:,win_size,:),1,1,depth,1);
                  derInputs{1} = derInputs{1}+inputs2.*derOutputs1;
                  [derInputs2,~] = vl_nnbilinearsampler(inputs{2},xxyy_s,inputs{1}.*derOutputs1);
                  derInputs{2} = derInputs{2}+derInputs2;
            end
        end         
        derParams = {} ;
    end

    function obj = Constraint_Correlation_Layer(varargin)
      obj.load(varargin);
    end  
    
    function initGrid(obj)
        xi = linspace(-1,1,obj.Ho);
        yi = linspace(-1,1,obj.Wo);
        [y,x] = meshgrid(xi,yi);
        x = x';
        y = y';
        x = single(x(:));
        y = single(y(:));
        obj.xx = y;
        obj.yy = x;
    end    
    
  end
  
end
