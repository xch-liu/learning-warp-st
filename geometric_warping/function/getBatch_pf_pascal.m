function inputs = getBatch_pf_pascal(imdb,batch)
    init_transform =  gpuArray(zeros(16,16,6,'single'));
    init_transform(:,:,1) = 1; init_transform(:,:,5) = 1;
    for i = 1:numel(batch)
        image1 = imread(fullfile(fullfile(imdb.images.data{batch(i)}),'image1.png'));
        image2 = imread(fullfile(fullfile(imdb.images.data{batch(i)}),'image2.png'));
        image1_mask = imread(fullfile(fullfile(imdb.images.data{batch(i)}),'mask1.png'));
        image1 = gpuArray(single(image1));
        image2 = gpuArray(single(image2)); 
        image1_mask = gpuArray(single(image1_mask)); 
        
        img1(:,:,:,i) = image1;
        img2(:,:,:,i) = image2;
        init(:,:,:,i) = init_transform;
        label(:,:,:,i) = image1_mask;
    end
    avim = gpuArray(single(cat(3,123.680,116.779,103.939)));
    img1 = bsxfun(@minus,img1,avim);
    img2 = bsxfun(@minus,img2,avim);
    inputs = {'label', label, 'init', init, 'f1_input', img1, 'f2_input', img2, 'img1', img1, 'img2', img2};   
end