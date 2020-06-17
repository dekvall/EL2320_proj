clear workspace;
close all
bad_computer = true;

if bad_computer
    video_name = "small/4-balls-high-small.mp4";
else
    video_name = "4-balls-high.mp4";
end
path = "../resx";
video_path = join([path, video_name],"/");
v = VideoReader(char(video_path));

green_lower = 50/360;
green_upper = 80/360;
while hasFrame(v)
    frame = readFrame(v);
    gray_frame = rgb2gray(frame);
    hsv_img = rgb2hsv(frame);
    h_channel = hsv_img(:,:,1);
    not_green_val = find(green_lower > h_channel | h_channel > green_upper);
    h_channel(not_green_val) = 0;
    
    not_green_val_sat = find(0.2 > hsv_img(:,:,2));
    gray_frame(not_green_val_sat) = 0;
    gray_frame(not_green_val) = 0;
    [centers, radii, metric] = imfindcircles(gray_frame,[15,30], 'Sensitivity', 0.95, 'EdgeThreshold', 0);
    imshow(gray_frame), hold on
    viscircles(centers, radii, 'EdgeColor', 'r')
    pause(0.15);
end