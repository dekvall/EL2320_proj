clear workspace;
video_name = "ball_img1.png";
path = "../resx";
video_path = join([path, video_name],"/");
v = imread(char(video_path));

green_lower = 50/360;
green_upper = 80/360;

frame = v;
gray_frame = rgb2gray(frame);
hsv_img = rgb2hsv(frame);
h_channel = hsv_img(:,:,1);
not_green_val = find(green_lower > h_channel | h_channel > green_upper);
h_channel(not_green_val) = 0;

not_green_val_sat = find(0.2 > hsv_img(:,:,2));
gray_frame(not_green_val_sat) = 0;
gray_frame(not_green_val) = 0;
gray_frame(gray_frame > 0) = 255;
[centers, radii, metric] = imfindcircles(gray_frame,[15,30], 'Sensitivity', 0.95, 'EdgeThreshold', 0)
hold on;
imshow(gray_frame)
viscircles(centers, radii, 'EdgeColor', 'b');       