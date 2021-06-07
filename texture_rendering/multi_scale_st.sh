# Script for multi scale texture style transfer

STYLE_IMAGE=examples/inputs/dali_melting_clock.png
CONTENT_IMAGE=examples/inputs/warped_clock.png

STYLE_WEIGHT=5e1
STYLE_SCALE=1.0
STYLE_WEIGHT2=2500 # Style weight for image size 2048 and above

GPU=0

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -print_iter 100 \
  -save_iter 1000 \
  -style_weight $STYLE_WEIGHT \
  -image_size 32 \
  -output_image out32.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -init image -init_image out32.png \
  -print_iter 100 \
  -save_iter 1000 \
  -style_weight $STYLE_WEIGHT \
  -image_size 64 \
  -output_image out64.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -init image -init_image out64.png \
  -print_iter 100 \
  -save_iter 1000 \
  -style_weight $STYLE_WEIGHT \
  -image_size 128 \
  -output_image out128.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -init image -init_image out128.png \
  -print_iter 100 \
  -save_iter 1000 \
  -style_weight $STYLE_WEIGHT \
  -image_size 256 \
  -output_image out256.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out256.png \
  -style_scale $STYLE_SCALE \
  -print_iter 100 \
  -save_iter 500 \
  -style_weight $STYLE_WEIGHT \
  -image_size 512 \
  -num_iterations 500 \
  -output_image out512.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

python neural_style.py \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out512.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -save_iter 200 \
  -style_weight $STYLE_WEIGHT \
  -image_size 1024 \
  -num_iterations 200 \
  -output_image out1024.png \
  -tv_weight 0 \
  -gpu $GPU \
  -backend cudnn

