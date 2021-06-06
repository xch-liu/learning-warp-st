function inputs = getBatch_pascal_synthetic(imdb,batch)
    rng('shuffle');

    %%
    scaling_factor_min = 0.75;
    scaling_factor_max = 1.25;
    rotation_angle_min = -pi/12;
    rotation_angle_max = +pi/12;
    shear_angle_min = -pi/12;
    shear_angle_max = +pi/12;
    translation_min = -0.4;
    translation_max = +0.4;
        
    tps_min = -0.1;
    tps_max = +0.1;    
    %%
       
    image_theta_affine = zeros(1,1,6,'single');
    image_theta_tps = zeros(1,1,18,'single');
    
	xi = linspace(-1,1,256);
	yi = linspace(-1,1,256);
	[yy,xx] = meshgrid(xi,yi);
    xx = xx';
    yy = yy';
	grid_ = permute(single(cat(3,yy,xx)),[3,1,2]);
    
	xi_tps = linspace(-1,1,3);
	yi_tps = linspace(-1,1,3);
	[yy_tps,xx_tps] = meshgrid(xi_tps,yi_tps);
    
    init_transform =  gpuArray(zeros(16,16,6,'single'));
    init_transform(:,:,1) = 1; init_transform(:,:,5) = 1;
    
    for i = 1:numel(batch)
        image1_raw = imread(fullfile(imdb.images.data{batch(i)},'image1.png'));
        image1_raw = single(imresize(image1_raw,[640,480]));
        image1_mask = ones(256,256,'single');
        if size(image1_raw,3) ~= 3, image1_raw = repmat(image1_raw,1,1,3); end
         
       %% Affine Transform
        scaling_factor_x = (scaling_factor_max-scaling_factor_min)*rand()+scaling_factor_min;
        scaling_factor_y = (scaling_factor_max-scaling_factor_min)*rand()+scaling_factor_min;
        rotation_angle = (rotation_angle_max-rotation_angle_min)*rand()+rotation_angle_min;
        shear_angle_x = (shear_angle_max-shear_angle_min)*rand()+shear_angle_min;
        shear_angle_y = (shear_angle_max-shear_angle_min)*rand()+shear_angle_min;
        translation_x = (translation_max-translation_min)*rand()+translation_min;
        translation_y = (translation_max-translation_min)*rand()+translation_min; 

        mat_scale = [scaling_factor_x,0;0,scaling_factor_y];
        mat_rotation = [cos(rotation_angle),sin(rotation_angle);-sin(rotation_angle),cos(rotation_angle)];
        mat_shear_x = [1,tan(shear_angle_x);0,1];
        mat_shear_y = [1,0;tan(shear_angle_y),1];
        A = mat_scale*mat_rotation*mat_shear_x*mat_shear_y;
        b = [translation_x;translation_y];
        A = A';
        
        image_theta_affine(1,1,1:4) = A(:);
        image_theta_affine(1,1,5:6) = b(:);
        
       %% TPS Transform
        yy_tps_rand = yy_tps(:)+(tps_max-tps_min)*rand(9,1)+tps_min;
        xx_tps_rand = xx_tps(:)+(tps_max-tps_min)*rand(9,1)+tps_min;
        
        image_theta_tps(1,1,1:9) = yy_tps_rand(:);
        image_theta_tps(1,1,1+9:9+9) = xx_tps_rand(:);
        
        % generate image A by cropping the raw image
        innerCropfactor = 9/16;
        image1 = image1_raw(round(640*(1-innerCropfactor)/2+1):end-round(640*(1-innerCropfactor)/2),...
            round(480*(1-innerCropfactor)/2+1):end-round(480*(1-innerCropfactor)/2),:);
        
        % add extra padding for enlarging the sampling region for image B
        paddingFactor = 1/2;
        image1_raw = imresize(image1_raw,[454 454]);  % delete line
        image1_raw = padarray(image1_raw, size(image1_raw(:,:,1))*paddingFactor, 'symmetric');   
        factor = paddingFactor*innerCropfactor;        
        
        tnf_affine = dagnn.AffineGridGenerator('Ho',256,'Wo',256); 
        samplingGrid_affine = tnf_affine.forward({image_theta_affine});
        
        tnf_tps = dagnnExtra.TpsGridGenerator('Ho',256,'Wo',256);        
        samplingGrid_tps = tnf_tps.forward({image_theta_tps});
        
        samplingGrid_new = samplingGrid_affine{1}+samplingGrid_tps{1}-grid_;     
        samplingGrid_flow = samplingGrid_affine{1}+samplingGrid_tps{1}-2*grid_;    
        bs = dagnn.BilinearSampler();
        image2 = bs.forward({image1_raw,samplingGrid_new*factor}); 
        image2 = image2{1};
        image2_mask = bs.forward({image1_mask,samplingGrid_new}); 
        image2_mask = image2_mask{1};
       
        image1_ = imresize(image1,[256,256]);
        image2_ = imresize(image2,[256,256]);
        image1 = image2_;
        image2 = image1_;        
        
        label_flow = imresize(permute(samplingGrid_flow,[2,3,1]),[32,32]).*repmat(imresize(single(image2_mask~=0),[32,32],'nearest'),1,1,2);  
        
        image1 = gpuArray(single(image1));
        image2 = gpuArray(single(image2));    
        
        img1(:,:,:,i) = image1;
        img2(:,:,:,i) = image2;
        init(:,:,:,i) = init_transform;
        label(:,:,:,i) = label_flow;
    end
    avim = gpuArray(single(cat(3,123.680,116.779,103.939)));
    img1 = bsxfun(@minus,img1,avim);
    img2 = bsxfun(@minus,img2,avim);
    inputs = {'label', label, 'init', init, 'f1_input', img1, 'f2_input', img2, 'img1', img1, 'img2', img2};   
end