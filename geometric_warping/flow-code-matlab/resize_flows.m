function [fo1, fo2] = resize_flows(fi1, fi2, newsize1, newsize2)
oldsize1 = size(fi1);
oldsize2 = size(fi2);

fo1 = fi1;
fo2 = fi2;

if oldsize1(1) ~= newsize1(1) || oldsize1(2) ~= newsize1(2) || oldsize2(1) ~= newsize2(1) || oldsize2(2) ~= newsize2(2)
    valid1 = imresize(bwdist(valid_flow_mask(fo1)), [newsize1(1) newsize1(2)], 'bilinear') == 0;
    valid2 = imresize(bwdist(valid_flow_mask(fo2)), [newsize2(1) newsize2(2)], 'bilinear') == 0;
    
    fo1 = imresize(fo1, [newsize1(1) newsize1(2)], 'bilinear');
    fo2 = imresize(fo2, [newsize2(1) newsize2(2)], 'bilinear');
    
    fo1 = resize_flow(fo1, oldsize1, oldsize2, newsize1, newsize2);
    fo2 = resize_flow(fo2, oldsize2, oldsize1, newsize2, newsize1);
    
    fo1 = set_flow_mask(fo1, valid1);
    fo2 = set_flow_mask(fo2, valid2);
end
    function [fo1] = resize_flow(fi1, oldsize1, oldsize2, newsize1, newsize2)
        [newX, newY] = meshgrid(1:newsize1(2), 1:newsize1(1));
        oldX = newX / newsize1(2) * oldsize1(2);
        oldY = newY / newsize1(1) * oldsize1(1);
        fo1(:,:,1) = (fi1(:,:,1) + oldX) / oldsize2(2) * newsize2(2) - newX;
        fo1(:,:,2) = (fi1(:,:,2) + oldY) / oldsize2(1) * newsize2(1) - newY;
    end

    function mask = valid_flow_mask(flow)
        mask = abs(flow(:,:,1)) < 1e9 & abs(flow(:,:,2)) < 1e9;
    end
    function flow = set_flow_mask(fi, mask)
        u = fi(:,:,1);
        v = fi(:,:,2);
        u(~mask) = 1e10;
        v(~mask) = 1e10;
        flow(:,:,1) = u;
        flow(:,:,2) = v;
    end
end