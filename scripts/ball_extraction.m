clear workspace;
video_name = "4-balls-good.mp4";
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
    %[centers, radii, metric] = imfindcircles(gray_frame,[10,50]);
    %viscircles(centers, radii, 'EdgeColor', 'b');       
    imshow(gray_frame)
    pause(1/v.FrameRate);
end