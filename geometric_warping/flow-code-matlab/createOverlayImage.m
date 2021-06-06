%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   This file is part of the HCI-Correspondence Estimation Benchmark.
%
%   More information on this benchmark can be found under:
%       http://hci.iwr.uni-heidelberg.de/Benchmarks/
%
%    Copyright (C) 2011  <Sellent, Lauer>
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function overlayImage = createOverlayImage( image, visImage )
%Create an overlay of the input image and visualization of the
%correspondences

%determine characteristics of input image
[height, width, channels] = size( image);
maxImage = double( max( image( : ) ) );

maxVis = 2^ceil( log2( double( max( visImage(:) ) ) ) );


%transfer input image to 3 channel rgb image with grayvalues in [0,1]
if ( 1 < channels )
    image = sum( double( image ), 3 )/channels;
else
    image = double( image );
end

image = image / maxImage; 
image = repmat( image, [1,1,3] );

%Add images for overlay
overlayImage = maxVis * image * 0.2 + double( visImage ) * 0.8;
overlayImage = cast( overlayImage, class( visImage ) );



    